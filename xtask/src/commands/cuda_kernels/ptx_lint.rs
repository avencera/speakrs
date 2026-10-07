//! Reject generic shared addresses narrowed without conversion to shared space

use std::collections::{BTreeMap, BTreeSet};

use color_eyre::eyre::{Result, bail};

/// Follow generic shared-address values within each PTX function
///
/// PTX register reuse and branches require a conservative fixed point: an address
/// that reaches a narrowing conversion on any path is unsafe; never carry taint
/// between functions, whose register names are independent
pub(super) fn check_shared_truncation(ptx: &str) -> Result<()> {
    let sites = shared_truncations(ptx);
    if !sites.is_empty() {
        bail!(
            "generic shared address reaches a 32-bit truncation; use cvta.to.shared before narrowing:\n    {}",
            sites.join("\n    ")
        );
    }

    Ok(())
}

fn shared_truncations(ptx: &str) -> Vec<String> {
    let code = without_comments(ptx);
    let tokens: Vec<_> = code
        .split(|c: char| c.is_whitespace() || "(),;{}:".contains(c))
        .filter(|token| !token.is_empty())
        .collect();
    let mut sites = Vec::new();
    let mut rest = code.as_str();
    for token in &tokens {
        if !matches!(*token, ".entry" | ".func") {
            continue;
        }

        let Some(start) = rest.find(token) else {
            break;
        };
        rest = &rest[start + token.len()..];
        // return parameters precede the name of a .func
        let declaration = rest.trim_start();
        let declaration = if declaration.starts_with('(') {
            declaration
                .split_once(')')
                .map_or("", |(_, tail)| tail)
                .trim_start()
        } else {
            declaration
        };
        let name = declaration
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("?");
        let Some(body_start) = rest.find('{') else {
            continue;
        };
        let before_body = &rest[..body_start];
        // an external declaration has no body; the brace belongs to another function
        if before_body.contains(';')
            || before_body.contains(".entry")
            || before_body.contains(".func")
        {
            continue;
        }

        let mut depth = 1;
        let tail = &rest[body_start + 1..];
        let Some(body_end) = tail.char_indices().find_map(|(offset, c)| {
            depth += i32::from(c == '{') - i32::from(c == '}');
            (depth == 0).then_some(offset)
        }) else {
            continue;
        };
        let instructions: Vec<_> = tail[..body_end]
            .split(';')
            .filter_map(Instruction::parse)
            .collect();
        // register reuse must not hide an earlier wide definition
        let mut widths = BTreeMap::new();
        for instruction in &instructions {
            if !instruction.has_destination() {
                continue;
            }
            for register in instruction.destinations() {
                let width = instruction.destination_width();
                widths
                    .entry(register)
                    .and_modify(|previous: &mut Option<usize>| {
                        *previous = (*previous).max(width);
                    })
                    .or_insert(width);
            }
        }
        let mut tainted = BTreeSet::new();
        for instruction in &instructions {
            if instruction.generic_shared() {
                tainted.extend(instruction.destinations());
            }
        }

        loop {
            let mut changed = false;
            for instruction in &instructions {
                if instruction.sanitizes() || !instruction.has_destination() {
                    continue;
                }
                if instruction.sources().any(|source| tainted.contains(source)) {
                    for destination in instruction.destinations() {
                        changed |= tainted.insert(destination);
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for instruction in &instructions {
            if instruction.generic_shared() && instruction.opcode.ends_with(".u32") {
                sites.extend(
                    instruction
                        .destinations()
                        .map(|register| format!("{name}: {register}")),
                );
                continue;
            }
            if instruction.truncates() {
                sites.extend(
                    instruction
                        .sources()
                        .filter(|source| {
                            tainted.contains(source)
                                && widths
                                    .get(source)
                                    .copied()
                                    .flatten()
                                    .is_none_or(|width| width > 32)
                        })
                        .map(|source| format!("{name}: {source}")),
                );
            }
        }
        rest = &tail[body_end + 1..];
    }

    sites
}

struct Instruction<'a> {
    opcode: &'a str,
    operands: Vec<Vec<&'a str>>,
}

impl<'a> Instruction<'a> {
    fn parse(statement: &'a str) -> Option<Self> {
        let statement = statement.trim();
        // declarations and block labels can precede the instruction in a statement
        let opcode = statement
            .split(|c: char| c.is_whitespace() || "{}:".contains(c))
            .find(|word| {
                !word.starts_with('.')
                    && (word.contains('.') || *word == "call")
                    && !word.starts_with('@')
            })?;
        let (_, operands) = statement.split_once(opcode)?;
        let is_call = opcode.split('.').next() == Some("call");
        let has_returns = !is_call || operands.trim_start().starts_with('(');
        let mut depth = 0;
        let mut operands: Vec<Vec<_>> = operands
            .split(|c| {
                if matches!(c, '{' | '(') {
                    depth += 1;
                }
                if matches!(c, '}' | ')') {
                    depth -= 1;
                }
                c == ',' && depth == 0
            })
            .map(|operand| {
                operand
                    .split(|c: char| c.is_whitespace() || "{}(),[]|+-".contains(c))
                    .filter(|word| {
                        word.starts_with('%')
                            || word.starts_with('_')
                            || word.starts_with(|c: char| c.is_ascii_alphabetic())
                    })
                    .collect()
            })
            .collect();
        // a call without return operands must not treat its callee as a destination
        if !has_returns {
            operands.insert(0, Vec::new());
        }
        Some(Self { opcode, operands })
    }

    fn generic_shared(&self) -> bool {
        self.opcode.starts_with("cvta.shared.")
    }

    fn sanitizes(&self) -> bool {
        self.opcode.starts_with("cvta.to.shared.")
    }

    fn has_destination(&self) -> bool {
        // memory writes and control flow have no register result
        !matches!(
            self.opcode.split('.').next(),
            Some("st" | "red" | "bra" | "brx" | "ret" | "exit")
        )
    }

    fn destinations(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.operands.first().into_iter().flatten().copied()
    }

    fn sources(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.operands.iter().skip(1).flatten().copied()
    }

    fn destination_width(&self) -> Option<usize> {
        let width = self.opcode.split('.').find_map(|part| {
            let (kind, digits) = part.split_at_checked(1)?;
            matches!(kind, "u" | "s" | "b" | "f")
                .then(|| digits.parse::<usize>().ok())
                .flatten()
        })?;
        Some(
            width
                / self
                    .operands
                    .first()
                    .map_or(1, |registers| registers.len().max(1)),
        )
    }

    fn truncates(&self) -> bool {
        if self.sanitizes() {
            return false;
        }
        let mut parts = self.opcode.split('.');
        if !matches!(parts.next(), Some("cvt" | "mov")) {
            return false;
        }
        let destination_type = parts.find(|part| {
            matches!(
                *part,
                "u32" | "s32" | "b32" | "f32" | "u64" | "s64" | "b64" | "f64"
            )
        });
        matches!(destination_type, Some("u32" | "s32" | "b32" | "f32"))
            || self
                .operands
                .first()
                .is_some_and(|registers| registers.len() > 1)
    }
}

fn without_comments(ptx: &str) -> String {
    let mut code = String::new();
    let mut chars = ptx.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '/' {
            code.push(c);
            continue;
        }
        match chars.peek() {
            Some('/') => {
                chars.next();
                for c in chars.by_ref() {
                    if c == '\n' {
                        code.push(c);
                        break;
                    }
                }
            }
            Some('*') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        break;
                    }
                }
                code.push(' ');
            }
            _ => code.push(c),
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::{check_shared_truncation, shared_truncations};

    #[test]
    fn pre_fix_lstm_has_all_eight_truncations() {
        let fixture = include_str!("../../../tests/fixtures/ptx/shared-truncation-lstm.ptx");
        let sites = shared_truncations(fixture);
        assert_eq!(
            sites,
            [
                "%rd69", "%rd79", "%rd94", "%rd104", "%rd115", "%rd126", "%rd143", "%rd145"
            ]
            .map(|register| format!("spk_lstm_projection_tf32: {register}"))
        );
        assert!(check_shared_truncation(fixture).is_err());
    }

    #[test]
    fn all_shared_propagation_and_truncation_paths_are_checked() {
        let fixture = include_str!("../../../tests/fixtures/ptx/shared-truncation-paths.ptx");
        let mut expected: Vec<_> = (0..8).map(|index| format!("path{index}: %out")).collect();
        expected.extend(
            [
                "shared32: %base",
                "signed: %base",
                "bits: %base",
                "move32: %base",
                "vector: %base",
                "vector: %wide",
                "backwards: %copy",
                "shared32_widened: %base",
                "shared32_widened: %wide",
                "named: %copy",
                "signed_mad: %out",
                "signed_mul: %out",
                "rzi: %base",
                "call_return: %out",
                "call_returns: %first",
                "call_returns: %second",
            ]
            .map(str::to_owned),
        );
        assert_eq!(shared_truncations(fixture), expected);
        assert!(check_shared_truncation(fixture).is_err());
    }

    #[test]
    fn converted_shared_addresses_and_function_local_registers_pass() {
        let fixture = include_str!("../../../tests/fixtures/ptx/shared-truncation-safe.ptx");
        assert!(check_shared_truncation(fixture).is_ok());
    }

    #[test]
    fn both_arithmetic_operands_predicates_and_transitive_moves_are_checked() {
        let fixture = ".visible .entry unsafe() {\ncvta.shared.u64 %base, smem;\nmov.b64 %copy, %base;\nadd.u64 %offset, 16, %copy;\nsub.s64 %address, 32, %offset;\n@%p cvt.u32.u64 %r, %address;\n}";
        assert_eq!(shared_truncations(fixture), ["unsafe: %address"]);
    }

    #[test]
    fn device_functions_and_external_declarations_are_isolated() {
        let fixture = ".extern .func external();\n.func (.param .b32 result) unsafe_func() { cvta.shared.u64 %base, smem; cvt.u32.u64 %r, %base; }\n.visible .entry safe() { mov.u64 %base, 1; cvt.u32.u64 %r, %base; }";
        assert_eq!(shared_truncations(fixture), ["unsafe_func: %base"]);
    }
}

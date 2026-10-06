//! Reject generic shared addresses narrowed without conversion to shared space

use std::collections::BTreeSet;

use color_eyre::eyre::{Result, bail};

/// Follow generic shared-address values within each PTX function
///
/// PTX register reuse and branches require a conservative fixed point: an address
/// that reaches a narrowing conversion on any path is unsafe. Never carry taint
/// between functions, whose register names are independent
pub(super) fn check_shared_truncation(ptx: &str) -> Result<()> {
    let sites = shared_truncations(ptx);
    if !sites.is_empty() {
        bail!(
            "generic shared address reaches cvt.u32.u64; use cvta.to.shared before narrowing:\n    {}",
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
        let instructions: Vec<Vec<_>> = tail[..body_end]
            .split(';')
            .map(|statement| {
                statement
                    .split(|c: char| c.is_whitespace() || ",{}:".contains(c))
                    .filter(|word| !word.is_empty())
                    .collect()
            })
            .collect();
        let mut tainted = BTreeSet::new();
        for words in &instructions {
            if let Some(op) = words.iter().position(|word| *word == "cvta.shared.u64")
                && let Some(destination) = words.get(op + 1)
            {
                tainted.insert(*destination);
            }
        }

        loop {
            let mut changed = false;
            for words in &instructions {
                let Some(op) = words.iter().position(|word| {
                    matches!(
                        *word,
                        "add.u64"
                            | "add.s64"
                            | "sub.u64"
                            | "sub.s64"
                            | "mov.u64"
                            | "mov.s64"
                            | "mov.b64"
                    )
                }) else {
                    continue;
                };
                let Some(destination) = words.get(op + 1) else {
                    continue;
                };
                if words[op + 2..]
                    .iter()
                    .any(|source| tainted.contains(source))
                {
                    changed |= tainted.insert(*destination);
                }
            }
            if !changed {
                break;
            }
        }

        for words in &instructions {
            if let Some(op) = words.iter().position(|word| *word == "cvt.u32.u64")
                && let Some(source) = words.get(op + 2).filter(|source| tainted.contains(*source))
            {
                sites.push(format!("{name}: {source}"));
            }
        }
        rest = &tail[body_end + 1..];
    }

    sites
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
        assert_eq!(sites.len(), 8, "{sites:?}");
        assert!(
            sites
                .iter()
                .all(|site| site.starts_with("spk_lstm_projection_tf32:"))
        );
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

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use color_eyre::eyre::{Context, Result, ensure, eyre};
use roxmltree::{Document, Node};
use serde::Serialize;

use super::store::atomic_write;

const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;

#[derive(Clone)]
struct XmlValue {
    text: String,
    formatted: Option<String>,
}

impl XmlValue {
    fn unsigned(&self, field: &str) -> Result<u64> {
        self.text
            .trim()
            .parse()
            .wrap_err_with(|| format!("invalid {field} value {:?}", self.text))
    }

    fn display(&self) -> &str {
        self.formatted.as_deref().unwrap_or(self.text.trim())
    }
}

#[derive(Serialize)]
pub(super) struct MetalTraceSummary {
    schema_version: u32,
    gpu_interval_table: PathBuf,
    command_submission_table: PathBuf,
    interval_count: usize,
    command_submission_count: usize,
    unique_command_buffer_count: usize,
    trace_span_seconds: f64,
    gpu_active_union_seconds: f64,
    gpu_active_fraction: f64,
    gpu_idle_gap_seconds: f64,
    summed_interval_seconds: f64,
    cpu_to_gpu_latency_median_seconds: f64,
    cpu_to_gpu_latency_p95_seconds: f64,
    interval_groups: BTreeMap<String, IntervalGroup>,
}

#[derive(Default, Serialize)]
struct IntervalGroup {
    count: usize,
    summed_seconds: f64,
}

struct Interval {
    start: u64,
    duration: u64,
    latency: u64,
    label: String,
    command_buffer: String,
}

pub(super) fn write_metal_summary(
    gpu_intervals: &Path,
    command_submissions: &Path,
    output: &Path,
) -> Result<()> {
    let intervals = read_gpu_intervals(gpu_intervals)?;
    ensure!(
        !intervals.is_empty(),
        "Metal trace has no GPU intervals in {}",
        gpu_intervals.display()
    );
    let command_submission_count = count_rows(command_submissions)?;
    let unique_command_buffer_count = intervals
        .iter()
        .map(|interval| interval.command_buffer.as_str())
        .filter(|id| !id.is_empty())
        .collect::<BTreeSet<_>>()
        .len();
    let first_start = intervals
        .iter()
        .map(|interval| interval.start)
        .min()
        .unwrap_or(0);
    let last_end = intervals
        .iter()
        .map(|interval| interval.start.saturating_add(interval.duration))
        .max()
        .unwrap_or(first_start);
    let trace_span = last_end.saturating_sub(first_start);
    let active_union = interval_union_duration(&intervals);
    let summed_duration = intervals
        .iter()
        .map(|interval| interval.duration)
        .sum::<u64>();
    let mut latencies = intervals
        .iter()
        .map(|interval| interval.latency)
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let mut groups = BTreeMap::<String, IntervalGroup>::new();
    for interval in &intervals {
        let group = groups
            .entry(label_group(&interval.label).to_owned())
            .or_default();
        group.count += 1;
        group.summed_seconds += seconds(interval.duration);
    }

    let summary = MetalTraceSummary {
        schema_version: 1,
        gpu_interval_table: gpu_intervals.to_owned(),
        command_submission_table: command_submissions.to_owned(),
        interval_count: intervals.len(),
        command_submission_count,
        unique_command_buffer_count,
        trace_span_seconds: seconds(trace_span),
        gpu_active_union_seconds: seconds(active_union),
        gpu_active_fraction: if trace_span == 0 {
            0.0
        } else {
            active_union as f64 / trace_span as f64
        },
        gpu_idle_gap_seconds: seconds(trace_span.saturating_sub(active_union)),
        summed_interval_seconds: seconds(summed_duration),
        cpu_to_gpu_latency_median_seconds: seconds(percentile(&latencies, 0.5)),
        cpu_to_gpu_latency_p95_seconds: seconds(percentile(&latencies, 0.95)),
        interval_groups: groups,
    };
    let mut bytes = serde_json::to_vec_pretty(&summary)?;
    bytes.push(b'\n');
    atomic_write(output, &bytes)
}

fn read_gpu_intervals(path: &Path) -> Result<Vec<Interval>> {
    let xml = fs::read_to_string(path)
        .wrap_err_with(|| format!("failed to read Metal trace table {}", path.display()))?;
    let document = Document::parse(&xml)
        .wrap_err_with(|| format!("invalid Metal trace XML in {}", path.display()))?;
    let values = collect_values(&document);
    let schema = schema_columns(&document)?;
    let index = |name: &str| {
        schema
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| eyre!("Metal trace schema is missing {name}"))
    };
    let start = index("start")?;
    let duration = index("duration")?;
    let latency = index("start-latency")?;
    let label = index("event-label")?;
    let command_buffer = index("cmdbuffer-id")?;
    let mut intervals = Vec::new();
    for row in document
        .descendants()
        .filter(|node| node.has_tag_name("row"))
    {
        let cells = row.children().filter(Node::is_element).collect::<Vec<_>>();
        ensure!(
            cells.len() == schema.len(),
            "Metal trace row has {} cells, expected {}",
            cells.len(),
            schema.len()
        );
        intervals.push(Interval {
            start: resolve(cells[start], &values)?.unsigned("start")?,
            duration: resolve(cells[duration], &values)?.unsigned("duration")?,
            latency: resolve(cells[latency], &values)?.unsigned("start-latency")?,
            label: resolve(cells[label], &values)?.display().to_owned(),
            command_buffer: resolve(cells[command_buffer], &values)?
                .display()
                .to_owned(),
        });
    }

    Ok(intervals)
}

fn collect_values(document: &Document<'_>) -> HashMap<String, XmlValue> {
    document
        .descendants()
        .filter_map(|node| {
            let id = node.attribute("id")?;
            Some((id.to_owned(), xml_value(node)))
        })
        .collect()
}

fn resolve(node: Node<'_, '_>, values: &HashMap<String, XmlValue>) -> Result<XmlValue> {
    if let Some(reference) = node.attribute("ref") {
        return values
            .get(reference)
            .cloned()
            .ok_or_else(|| eyre!("Metal trace contains unresolved ref {reference}"));
    }

    Ok(xml_value(node))
}

fn xml_value(node: Node<'_, '_>) -> XmlValue {
    XmlValue {
        text: node.text().unwrap_or_default().to_owned(),
        formatted: node.attribute("fmt").map(ToOwned::to_owned),
    }
}

fn schema_columns(document: &Document<'_>) -> Result<Vec<String>> {
    let schema = document
        .descendants()
        .find(|node| node.has_tag_name("schema"))
        .ok_or_else(|| eyre!("Metal trace XML has no schema"))?;
    let columns = schema
        .children()
        .filter(|node| node.has_tag_name("col"))
        .map(|column| {
            column
                .descendants()
                .find(|node| node.has_tag_name("mnemonic"))
                .and_then(|node| node.text())
                .map(ToOwned::to_owned)
                .ok_or_else(|| eyre!("Metal trace schema column has no mnemonic"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!columns.is_empty(), "Metal trace schema has no columns");

    Ok(columns)
}

fn count_rows(path: &Path) -> Result<usize> {
    let xml = fs::read_to_string(path)
        .wrap_err_with(|| format!("failed to read Metal trace table {}", path.display()))?;
    let document = Document::parse(&xml)
        .wrap_err_with(|| format!("invalid Metal trace XML in {}", path.display()))?;

    Ok(document
        .descendants()
        .filter(|node| node.has_tag_name("row"))
        .count())
}

fn interval_union_duration(intervals: &[Interval]) -> u64 {
    let mut ranges = intervals
        .iter()
        .map(|interval| {
            (
                interval.start,
                interval.start.saturating_add(interval.duration),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    let Some(&(mut start, mut end)) = ranges.first() else {
        return 0;
    };
    let mut total = 0u64;
    for &(next_start, next_end) in &ranges[1..] {
        if next_start > end {
            total = total.saturating_add(end.saturating_sub(start));
            start = next_start;
            end = next_end;
        } else {
            end = end.max(next_end);
        }
    }

    total.saturating_add(end.saturating_sub(start))
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let index = ((sorted.len().saturating_sub(1)) as f64 * quantile).round() as usize;
    sorted.get(index).copied().unwrap_or(0)
}

fn label_group(label: &str) -> &'static str {
    if label.contains("segmentation-") {
        "segmentation"
    } else if label.contains("wespeaker-fbank") {
        "filterbank"
    } else if label.contains("wespeaker-chunk-emb") {
        "chunk-embedding"
    } else {
        "other"
    }
}

fn seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / NANOSECONDS_PER_SECOND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_union_does_not_double_count_overlap() {
        let intervals = [
            interval(10, 10),
            interval(15, 10),
            interval(30, 5),
            interval(34, 3),
        ];

        assert_eq!(interval_union_duration(&intervals), 22);
    }

    #[test]
    fn label_group_uses_stable_model_families() {
        assert_eq!(label_group("segmentation-3.0-b64_main"), "segmentation");
        assert_eq!(label_group("wespeaker-fbank_main"), "filterbank");
        assert_eq!(label_group("wespeaker-chunk-emb-p1s"), "chunk-embedding");
        assert_eq!(label_group("GPU Execution"), "other");
    }

    fn interval(start: u64, duration: u64) -> Interval {
        Interval {
            start,
            duration,
            latency: 0,
            label: String::new(),
            command_buffer: String::new(),
        }
    }
}

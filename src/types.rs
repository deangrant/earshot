//! Result types produced by transcription.

/// A timed text segment from a transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Segment start time in seconds.
    pub start: f64,
    /// Segment end time in seconds.
    pub end: f64,
    /// Transcribed text for this segment.
    ///
    /// Leading and trailing whitespace is trimmed when building a
    /// [`TranscriptionResult`].
    pub text: String,
}

impl Segment {
    /// Creates a segment from Whisper centisecond timestamps.
    ///
    /// Whisper reports times in centiseconds (1/100 s). This converts them to
    /// seconds for the public API.
    pub fn from_centiseconds(start_cs: i64, end_cs: i64, text: impl Into<String>) -> Self {
        Self {
            start: start_cs as f64 / 100.0,
            end: end_cs as f64 / 100.0,
            text: text.into(),
        }
    }
}

/// Full transcription output including language and timed segments.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    /// Detected or configured language code (for example `"en"`).
    pub language: String,
    /// Space-joined transcript from all non-empty trimmed segment texts.
    pub text: String,
    /// Timed segments in order (segment text is trimmed).
    pub segments: Vec<Segment>,
}

impl TranscriptionResult {
    /// Builds a result from a language code and ordered segments.
    ///
    /// Each segment’s text is trimmed. [`Self::text`] joins non-empty trimmed
    /// segments with a single space. Whitespace-only segments stay in
    /// [`Self::segments`] with empty text so timestamps are preserved.
    pub fn from_segments(language: impl Into<String>, mut segments: Vec<Segment>) -> Self {
        for seg in &mut segments {
            seg.text = seg.text.trim().to_owned();
        }
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Self {
            language: language.into(),
            text,
            segments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_centiseconds_to_seconds() {
        let seg = Segment::from_centiseconds(150, 320, "hello");
        assert!((seg.start - 1.5).abs() < f64::EPSILON);
        assert!((seg.end - 3.2).abs() < f64::EPSILON);
        assert_eq!(seg.text, "hello");
    }

    #[test]
    fn joins_segment_text() {
        let result = TranscriptionResult::from_segments(
            "en",
            vec![
                Segment::from_centiseconds(0, 100, " Hello "),
                Segment::from_centiseconds(100, 200, "world"),
            ],
        );
        assert_eq!(result.language, "en");
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "Hello");
        assert_eq!(result.segments[1].text, "world");
    }

    #[test]
    fn whitespace_only_segment_kept_but_omitted_from_text() {
        let result = TranscriptionResult::from_segments(
            "en",
            vec![
                Segment::from_centiseconds(0, 100, "Hello"),
                Segment::from_centiseconds(100, 150, "   "),
                Segment::from_centiseconds(150, 200, "world"),
            ],
        );
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.segments.len(), 3);
        assert_eq!(result.segments[1].text, "");
        assert!((result.segments[1].start - 1.0).abs() < f64::EPSILON);
        assert!((result.segments[1].end - 1.5).abs() < f64::EPSILON);
    }
}

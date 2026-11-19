
class TextChunkingService:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_overlap_text(self, text: str) -> str:

        if self.chunk_overlap == 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        overlap_text = text[-self.chunk_overlap:]

        # Move to the beginning of the next complete word
        first_space = overlap_text.find(" ")

        if first_space == -1:
            return ""

        return overlap_text[first_space + 1:]

    def _split_text(self, text: str) -> list[str]:

        words = text.split()

        text_parts = []
        current_part = ""

        for word in words:

            candidate = (
                current_part + " " + word
                if current_part
                else word
            )

            if len(candidate) > self.chunk_size and current_part:
                text_parts.append(current_part)
                current_part = word

            else:
                current_part = candidate

        if current_part:
            text_parts.append(current_part)

        return text_parts

    def _get_overlap_segments(
        self,
        segments: list[dict],
        overlap_text: str
    ) -> list[dict]:

        if not overlap_text:
            return []

        contributing_segments = []
        accumulated_text = ""

        for segment in reversed(segments):

            segment_text = segment["text"].strip()

            accumulated_text = (
                segment_text + " " + accumulated_text
                if accumulated_text
                else segment_text
            )

            contributing_segments.insert(0, segment)

            # Stop once we've collected enough text
            if len(accumulated_text.strip()) >= len(overlap_text):
                break

        return contributing_segments

    def chunk_transcript(self, transcript: dict) -> list[dict]:

        segments = transcript.get("segments", [])

        # 1. Convert transcript segments into timestamped word tokens
        word_tokens = []

        for segment in segments:

            segment_text = segment["text"].strip()

            if not segment_text:
                continue

            # Use the existing helper to split text into size-limited parts
            text_parts = self._split_text(segment_text)

            for part in text_parts:

                for word in part.split():

                    # Handle a single word larger than chunk_size
                    if len(word) > self.chunk_size:
                        word_parts = [
                            word[i:i + self.chunk_size]
                            for i in range(0, len(word), self.chunk_size)
                        ]
                    else:
                        word_parts = [word]

                    for word_part in word_parts:
                        word_tokens.append({
                            "text": word_part,
                            "start": segment["start"],
                            "end": segment["end"]
                        })

        # 2. Build chunks from the word tokens
        chunks = []
        cursor = 0

        while cursor < len(word_tokens):

            current_tokens = []
            current_text = ""

            while cursor + len(current_tokens) < len(word_tokens):

                token = word_tokens[cursor + len(current_tokens)]

                candidate_text = (
                    current_text + " " + token["text"]
                    if current_text
                    else token["text"]
                )

                if len(candidate_text) > self.chunk_size:
                    break

                current_tokens.append(token)
                current_text = candidate_text

            # 3. Save the current chunk
            chunks.append({
                "chunk_index": len(chunks),
                "text": current_text,
                "start": current_tokens[0]["start"],
                "end": current_tokens[-1]["end"]
            })

            # 4. Find whole-word overlap tokens
            next_cursor = cursor + len(current_tokens)

            # Stop if all transcript tokens have been processed
            if next_cursor >= len(word_tokens):
                break

            overlap_tokens = []
            overlap_length = 0

            if self.chunk_overlap > 0 and len(current_tokens) > 1:

                for token in reversed(current_tokens):

                    candidate_length = (
                        overlap_length + len(token["text"]) + 1
                        if overlap_tokens
                        else len(token["text"])
                    )

                    # Never carry the entire chunk into the next chunk
                    if (
                        candidate_length <= self.chunk_overlap
                        and len(overlap_tokens) < len(current_tokens) - 1
                    ):
                        overlap_tokens.append(token)
                        overlap_length = candidate_length
                    else:
                        break

            # 5. Move the cursor, retaining overlap tokens
            cursor = next_cursor - len(overlap_tokens)

        return chunks

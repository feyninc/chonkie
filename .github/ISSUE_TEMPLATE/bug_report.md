---
name: Bug report
about: Chonkie did something wrong
title: "Bug:"
labels: bug
assignees: shreyash-chonkie, chonk-lain, chonknick

---

<!-- Please write this in your own words. See AI_POLICY.md. AI-written reports with no reproduction get closed. -->

**What happened**
What you expected, and what Chonkie did instead.

**To reproduce**
The smallest input text and code (or CLI command, or HTTP request) that shows the bug.

```
paste it here
```

**Which guarantee broke?** (if any)
- [ ] Joining the chunks doesn't give back the original text
- [ ] `text[start:end]` doesn't match the chunk's text
- [ ] `token_count` is wrong or exceeds `chunk_size`
- [ ] Same input gave different chunks
- [ ] Spaces or delimiters dropped, or a Unicode character split

**Environment**
- Chonkie version:
- Where: Python / JS/TS / CLI / server / Docker
- Chunker and its settings:
- Tokenizer:
- OS:

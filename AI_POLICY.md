# AI Policy

This policy applies to everyone who works on Chonkie: outside contributors and maintainers alike.

AI-assisted coding is allowed. The bar for what lands does not change. Every change must pass the correctness rules in [CONTRIBUTING.md](CONTRIBUTING.md), and whoever opens the pull request is responsible for every line in it, however it was written. Maintainers are responsible for everything that ships in a release.

## Humans open pull requests and write comments

**AI must not open pull requests or issues, or write comments.** A person opens every pull request under their own account, has read the whole diff, and can explain it. Pull requests, issues, and comments that appear to be opened or written by a bot or an agent will be closed or hidden, possibly without notice.

This applies to maintainers too: if an agent wrote the code, a maintainer reviews it, opens the pull request as themselves, and answers for it.

- **Issues:** describe the problem in your own words. Include the input, what you expected, and what Chonkie did. An AI-written bug report with no reproduction will be closed.
- **Pull requests:** the description and your answers to review questions are yours. Do not paste AI responses in reply to questions.

## Disclose the prompts

If AI wrote a significant part of a pull request, the description must include the prompts you used, in a collapsed `<details>` block under an **AI prompts** heading. "Significant" means more than autocomplete or a few lines: an agent session, a generated function, a generated test suite. Paste the prompts as you wrote them, not a summary. Leave out secrets and private data.

Small assists (autocomplete, a renamed variable, a one-line suggestion) need no disclosure.

## Quoting AI

If you want to include something an AI said, put it in a quote block (`>`), say it came from an AI, and add your own note on why it matters. Keep it short.

## Writing in a second language

AI is useful if English isn't your first language. If you use it to edit your writing, check that the result still says what you mean, in your voice. If you use it to translate, write in your own language and include the translation in a quote block.

---

Adapted from [Jiff's AI policy](https://github.com/BurntSushi/jiff/blob/master/AI_POLICY.md), which was adapted from [uv's](https://github.com/astral-sh/.github/blob/c5187e200db51bfe11d56e13053d29bd3793fdd8/AI_POLICY.md).

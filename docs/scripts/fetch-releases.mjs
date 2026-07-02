import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const OUT_DIR = path.resolve('public', 'data');

const REPOS = [
  { repo: 'chonkie-inc/chonkie', file: 'releases.json' },
];

async function fetchReleases(repo, outFile) {
  console.log(`[Releases] Fetching releases from ${repo}...`);

  const res = await fetch(`https://api.github.com/repos/${repo}/releases?per_page=50`, {
    headers: {
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'chonkie-docs',
      ...(process.env.GITHUB_TOKEN && { 'Authorization': `token ${process.env.GITHUB_TOKEN}` }),
    },
  });

  if (!res.ok) {
    console.error(`[Releases] Failed to fetch ${repo}: ${res.status} ${res.statusText}`);
    console.error('[Releases] Skipping (using cached data if available)');
    return;
  }

  const releases = await res.json();

  const simplified = releases.map((r) => ({
    id: r.id,
    name: r.name || r.tag_name,
    tag: r.tag_name,
    published_at: r.published_at,
    body: r.body || '',
    html_url: r.html_url,
  }));

  const outPath = path.join(OUT_DIR, outFile);
  await writeFile(outPath, JSON.stringify(simplified, null, 2));
  console.log(`[Releases] Saved ${simplified.length} releases to ${outPath}`);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  await Promise.all(REPOS.map(({ repo, file }) => fetchReleases(repo, file)));
}

main().catch((err) => {
  console.error('[Releases] Error:', err.message);
  console.error('[Releases] Skipping release sync');
});

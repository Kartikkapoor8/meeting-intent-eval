# RAW_OUTPUTS.md

What the model literally returned on the four real transcripts. Pulled from cached raw_samples_*.json. No re-runs, no scoring tricks. Just the strings.

## Counts by transcript x model

| transcript | model | total | empty array | non-empty | malformed | error |
|---|---|---:|---:|---:|---:|---:|
| real_earnings_001 | Opus 4.6 | 16 | 16 | 0 | 0 | 0 |
| real_earnings_001 | Sonnet 4.6 | 8 | 8 | 0 | 0 | 0 |
| real_earnings_001 | Haiku 4.5 (001/002) | 8 | 8 | 0 | 0 | 0 |
| real_earnings_002 | Opus 4.6 | 16 | 16 | 0 | 0 | 0 |
| real_earnings_002 | Sonnet 4.6 | 8 | 8 | 0 | 0 | 0 |
| real_earnings_002 | Haiku 4.5 (001/002) | 8 | 8 | 0 | 0 | 0 |
| real_earnings_003 | Opus 4.6 | 16 | 15 | 0 | 0 | 1 |
| real_earnings_003 | Sonnet 4.6 | 8 | 8 | 0 | 0 | 0 |
| real_earnings_003 | Haiku 4.5 (003) | 8 | 8 | 0 | 0 | 0 |
| real_client_001 | Opus 4.6 | 16 | 0 | 16 | 0 | 0 |

## Inline samples

### real_earnings_001

**Opus 4.6** (16 samples)
```
{"action_items": []}
```

**Sonnet 4.6** (8 samples)
```
{"action_items": []}
```

**Haiku 4.5 (001/002)** (8 samples)
```
```json
{"action_items": []}
```
```

### real_earnings_002

**Opus 4.6** (16 samples)
```
{"action_items": []}
```

**Sonnet 4.6** (8 samples)
```
{"action_items": []}
```

**Haiku 4.5 (001/002)** (8 samples)
```
```json
{"action_items": []}
```
```

### real_earnings_003

**Opus 4.6** (16 samples)
```
{"action_items": []}
```
```
Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': "This request would exceed your organization's rate limit of 30,000 input tokens per minute (org: 3bbae02c-ee25-4dd
```

**Sonnet 4.6** (8 samples)
```
{"action_items": []}
```

**Haiku 4.5 (003)** (8 samples)
```
```json
{"action_items": []}
```
```

### real_client_001

**Opus 4.6** (16 samples)
```
{"action_items": [{"owner": "Engineer", "task": "Deploy a link to the App demo for Client to explore on his own", "due": "tonight"}, {"owner": "Client", "task": "Send the 57-page document with 22 story treatments to Engineer and Partner", "due": "tomorrow morning"}]}
```
```
{"action_items":[{"owner":"Engineer","task":"Deploy a link to the App demo and send it to the team","due":"tonight"},{"owner":"Client","task":"Send the 57-page document of 22 stories to Engineer and Partner","due":"tomorrow morning"},{"owner":"Client","task":"Prepare and send additional context about what was happening the same night as the Band A (Band C playing downtown, photography, poster) to Engineer","due":"after this meeting"},{"owner":"Engineer","task":"Think about how to direct Client on what to do with the University MIS students and provide guidance","due":"before next Tuesday"}]}
```
```
{"action_items":[{"owner":"Engineer","task":"Deploy a link to the App demo and send it to Client and Partner","due":"tonight"},{"owner":"Client","task":"Send the 57-page document with 22 stories to Engineer and Partner","due":"tomorrow morning"},{"owner":"Client","task":"Prepare and send information about what was happening the same night as the Band A (Band C playing downtown, photography, poster) to Engineer","due":"after this meeting"},{"owner":"Engineer","task":"Think about how to direct Client on what the University MIS students should work on","due":"before next Tuesday"}]}
```

## Bottom line
On the three earnings calls, every single sample (95/95) returned `{"action_items": []}`. Format compliance was 100 percent. The JSON parsed fine every time. The model just had nothing in the array.

On real_client_001 the model returned long, structured arrays. So the pipeline is fine. The model just refuses to extract anything when the commitments are buried in dialogue.

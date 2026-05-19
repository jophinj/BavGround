# Dataset README

This folder contains the BavGround benchmark data used in the paper.

## Files

Each JSON file corresponds to one cultural category:

| File | Category | Source questions | Instances |
| --- | --- | ---: | ---: |
| `arts_and_identity.json` | Arts and Identity | 25 | 75 |
| `building_and_sacred_heritage.json` | Building and Sacred Heritage | 29 | 87 |
| `culinary.json` | Culinary | 25 | 75 |
| `historical.json` | Historical | 25 | 75 |
| `landscape.json` | Landscape | 25 | 75 |
| `language.json` | Bavarian Dialect and Language | 25 | 75 |
| `living_traditions_and_customs.json` | Living Traditions and Customs | 27 | 81 |
| `politics.json` | Politics | 25 | 75 |

Total: 206 source questions and 618 language-specific instances.

## Schema

Each file has this shape:

```json
{
  "version": "0.1",
  "items": [
    {
      "id": "arts_and_identity_en_001",
      "lang": "en",
      "type": "mcq",
      "prompt": "...",
      "options": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      },
      "gold": "B",
      "meta": {
        "bias": "cultural",
        "category": "arts_and_identity",
        "source": "..."
      }
    }
  ]
}
```

The `source` field appears only for grounded items.

## Item IDs

Item IDs follow:

```text
<category>_<language>_<number>
```

Examples:

- `language_bar_015`
- `politics_de_021`
- `arts_and_identity_en_001`

Languages:

- `en`: English
- `de`: German
- `bar`: Bavarian

## GEN and GRD Subsets

The paper uses two question subsets:

- `001` to `010`: general-knowledge items, abbreviated `GEN`
- later item numbers: source-grounded regional items, abbreviated `GRD`

For example, `culinary_en_004` is `GEN`, while `culinary_en_014` is `GRD`.

## Source Metadata

Grounded questions include source references under `item.meta.source`. These
references include public news URLs and book/article references used for
validation. The source text itself is not redistributed here.

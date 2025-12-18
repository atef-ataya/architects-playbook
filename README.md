# The Architect's Playbook - Code Repository

**5 Pillars of Production AI Agent Architecture**

This repository contains all code examples from the book "The Architect's Playbook" by Ataya.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/atefataya/architects-playbook.git
cd architects-playbook

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run example
python examples/complete_pipeline.py
```

## Repository Structure

```
├── pillar_1_mcp/           # MCP connection pool and tools
├── pillar_2_visual_ai/     # Visual AI automation
├── pillar_3_langgraph/     # Multi-agent orchestration
├── pillar_4_flight_deck/   # Execution governance
├── pillar_5_testing/       # Testing strategies
├── bonus_storm_gateway/    # Container orchestration
├── examples/               # Complete working examples
├── tests/                  # Test suites
└── config/                 # Configuration management
```

## Requirements

- Python 3.11+
- OpenAI API key
- (Optional) LangSmith API key for tracing

## License

MIT License - See LICENSE file for details.

## Author

**Ataya**
- YouTube: [youtube.com/@atefataya](https://youtube.com/@atefataya)
- Website: [atefataya.com](https://atefataya.com)
- Twitter: [@atayatef](https://twitter.com/atayatef)

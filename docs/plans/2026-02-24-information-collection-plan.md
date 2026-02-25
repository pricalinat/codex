# Information Collection Plan for Multi-Agent System

## Objective
Build a reusable collection pipeline that captures, normalizes, evaluates, and stores multi-source information for downstream agent tasks.

## Collection Scope
- Web/news sources
- Social/community feeds
- Internal files/datasets
- User-uploaded structured files

## Multi-Agent Roles
- PlannerAgent: turns user intent into collection tasks and priority.
- SourceDiscoveryAgent: finds candidate sources and de-duplicates them.
- CollectorAgent: fetches content and captures metadata.
- NormalizerAgent: extracts structured fields and canonical schema.
- EvaluatorAgent: scores relevance, freshness, and trust.
- SynthesizerAgent: generates concise insight summaries.
- ReviewerAgent: validates quality and flags risky claims.

## Shared Schema
- `source_id`: unique source identifier
- `uri`: canonical URL/path
- `collected_at`: UTC timestamp
- `published_at`: source publication time
- `content_type`: article/post/document
- `language`: ISO language code
- `raw_content`: original payload
- `normalized`: extracted entities, tags, and key facts
- `scores`: relevance/freshness/trust
- `trace`: agent/action/tool audit trail

## Pipeline
1. Planner defines query intent and constraints.
2. Discovery expands and ranks candidate sources.
3. Collector fetches and stores raw snapshots.
4. Normalizer extracts structure and key fields.
5. Evaluator computes ranking and confidence.
6. Synthesizer builds summaries and evidence references.
7. Reviewer approves output or requests retries.

## Reliability Controls
- Retry policy by source type.
- Circuit breaker for unstable endpoints.
- Source-level cooldown and rate limiting.
- Human approval gate for low-confidence outputs.

## Metrics
- Throughput: collected items/min
- Freshness latency: publish->collect delay
- Precision proxy: reviewer acceptance rate
- Coverage: unique sources per intent
- Cost: tokens/tool invocations per task

## Local Rollout Sequence
- v1: in-memory pipeline + deterministic tests
- v2: persistent state + async queue
- v3: provider adapters + approval dashboard

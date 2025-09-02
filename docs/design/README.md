# vLLM Architecture Documentation Index

This directory contains comprehensive technical architecture documentation for vLLM, including detailed Mermaid diagrams that illustrate the system's design and components.

## Architecture Documents

### 📋 [Technical Architecture Overview](technical_architecture.md)
**NEW** - Comprehensive technical architecture documentation with 10 detailed Mermaid diagrams covering:

- **System Overview** - Complete system architecture with all major components
- **Core Components** - Internal architecture and component interactions  
- **Request Processing Flow** - Sequence diagrams showing request lifecycle
- **Distributed Architecture** - Multi-node and parallelism strategies
- **Memory Management** - PagedAttention and KV cache architecture
- **Class Hierarchy** - Object relationships and dependencies
- **Data Flow** - Step-by-step inference pipeline
- **API Integration Points** - Available APIs and interfaces
- **Deployment Options** - Various deployment configurations
- **Performance Optimizations** - Key performance features

### 📋 [Architecture Overview](arch_overview.md) 
**ENHANCED** - Original architecture overview enhanced with 4 additional Mermaid diagrams:

- **Entrypoints** - User interfaces and API entry points
- **LLM Engine** - Core engine components and processing flow
- **Worker Architecture** - Multi-GPU worker setup and responsibilities
- **Model Runner** - Model execution and initialization flow

### Other Design Documents

- [PagedAttention Design](paged_attention.md) - Memory-efficient attention mechanism
- [Multiprocessing Design](multiprocessing.md) - Process management and communication
- [Distributed P2P Connector](p2p_nccl_connector.md) - Inter-node communication
- [Prefix Caching](prefix_caching.md) - Shared prompt optimization
- [Plugin System](plugin_system.md) - Extensibility framework
- [Metrics System](metrics.md) - Performance monitoring and observability
- [HuggingFace Integration](huggingface_integration.md) - Model loading and compatibility
- [Torch Compile](torch_compile.md) - PyTorch compilation optimizations
- [Fused MoE Kernel](fused_moe_modular_kernel.md) - Mixture-of-Experts optimization
- [Multimodal Processing](mm_processing.md) - Vision-language model support

## Diagram Types Used

The architecture documentation uses several types of Mermaid diagrams:

- **📊 Flowcharts** - System architecture and component relationships
- **📈 Sequence Diagrams** - Request processing and interaction flows  
- **🏗️ Class Diagrams** - Object relationships and inheritance
- **🧠 Mind Maps** - Feature categorization and optimization strategies
- **📋 Graph Diagrams** - Network topologies and communication patterns

## Viewing the Documentation

### Local Development
To view the documentation locally with properly rendered Mermaid diagrams:

```bash
# Install documentation dependencies
pip install -r requirements/docs.txt

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Online Documentation
The documentation is automatically built and deployed to the official vLLM documentation site with full Mermaid diagram support.

## Contributing to Architecture Documentation

When adding or updating architecture documentation:

1. Use Mermaid diagrams to illustrate complex concepts
2. Follow the established styling patterns for consistency
3. Include both high-level overviews and detailed component diagrams
4. Test diagrams locally before committing
5. Update this index when adding new documents

## Mermaid Diagram Guidelines

- Use consistent color schemes and styling
- Include descriptive labels and annotations
- Group related components in subgraphs
- Use appropriate diagram types for the content
- Ensure diagrams are readable at different screen sizes
- Test in both light and dark themes
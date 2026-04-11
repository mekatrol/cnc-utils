"""Core package for gerber2nc.

The package is split by manufacturing concern:
- Gerber parsers recover copper geometry and the mechanical board outline.
- The drill parser reads Excellon hole data.
- Geometry helpers convert preserved copper into isolation-routing paths.
- Visualization and G-code generation turn those derived paths into an
  inspectable and machinable workflow.
"""

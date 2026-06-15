"""Taxonomy loaders.

Load and normalize the global taxonomies into the knowledge graph:

    - ESCO      : EU knowledge, skills, and competencies
    - O*NET     : U.S. Department of Labor occupations
    - SFIA      : global framework for digital skills and competencies
    - BLS       : Occupational Outlook Handbook (career guidance)
    - Lightcast : crowd-sourced common skills language

Skeleton — each loader to be added (see ROADMAP deliverable 5).
"""

#: Taxonomies in scope for the project.
SUPPORTED_TAXONOMIES = ("ESCO", "ONET", "SFIA", "BLS", "Lightcast")

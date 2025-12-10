"""
Category Theory Module for GNSP.

This module provides category-theoretic abstractions for network security analysis:
- Base categorical structures (objects, morphisms, categories)
- Protocol categories modeling network protocol state machines
- Functors mapping between categories for security analysis
- Natural transformations for comparing security views
- Sheaves for distributed/local-to-global anomaly detection

The categorical perspective enables compositional reasoning about
protocol behaviors and provides a rigorous framework for detecting
anomalies as violations of functorial properties.
"""

from gnsp.category.base import (
    Object,
    Morphism,
    Category,
    ProductCategory,
    OppositeCategory,
)
from gnsp.category.protocol import (
    ProtocolState,
    ProtocolTransition,
    ProtocolCategory,
    TCPCategory,
    HTTPCategory,
    DNSCategory,
)
from gnsp.category.functor import (
    Functor,
    IdentityFunctor,
    CompositionFunctor,
    ForgetfulFunctor,
    SecurityFunctor,
    TrafficFunctor,
)
from gnsp.category.transformation import (
    NaturalTransformation,
    IdentityTransformation,
    VerticalComposition,
    HorizontalComposition,
)
from gnsp.category.sheaf import (
    Site,
    Presheaf,
    Sheaf,
    SheafSection,
    NetworkSheaf,
    AnomalySheaf,
    GluingCondition,
)

__all__ = [
    # Base category
    "Object",
    "Morphism",
    "Category",
    "ProductCategory",
    "OppositeCategory",
    # Protocol categories
    "ProtocolState",
    "ProtocolTransition",
    "ProtocolCategory",
    "TCPCategory",
    "HTTPCategory",
    "DNSCategory",
    # Functors
    "Functor",
    "IdentityFunctor",
    "CompositionFunctor",
    "ForgetfulFunctor",
    "SecurityFunctor",
    "TrafficFunctor",
    # Natural transformations
    "NaturalTransformation",
    "IdentityTransformation",
    "VerticalComposition",
    "HorizontalComposition",
    # Sheaves
    "Site",
    "Presheaf",
    "Sheaf",
    "SheafSection",
    "NetworkSheaf",
    "AnomalySheaf",
    "GluingCondition",
]

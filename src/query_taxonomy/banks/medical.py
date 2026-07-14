from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    AmbiguityTier,
    Domain,
    RegexBank,
    StructuralIdentifier,
)

_UPPER_ALNUM = RegexBuilder().any_of().range("A", "Z").range("0", "9").end()


class MedicalCodeBank(RegexBank):
    """ICD-10 with decimal (J45.909 — bare J45 excluded as too generic),
    CAS numbers (50-00-0), SNP ids (rs429358)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.MEDICAL_CODE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .range("A", "Z")
                    .exactly(2).digit()
                    .char(".")
                    .between(1, 4).digit()
                .end()
                .group()
                    .between(2, 7).digit()
                    .char("-")
                    .exactly(2).digit()
                    .char("-")
                    .digit()
                .end()
                .group()
                    .string("rs")
                    .at_least(3).digit()
                .end()
            .end()
            .word_boundary()
        )


class ClinicalCodingBank(RegexBank):
    """Keyword-gated CPT / HCPCS / SNOMED / LOINC / DRG codes."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CLINICAL_CODING

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .string("CPT").string("HCPCS").string("SNOMED CT").string("SNOMED")
                .string("LOINC").string("DRG")
            .end()
            .optional().whitespace_char()
            .digit()
            .between(2, 9).any_of().range("0", "9").char("-").end()
            .word_boundary()
        )


class DrugIdBank(RegexBank):
    """Keyword-gated NDC and ATC drug identifiers. Bare ATC codes (A10BA02)
    without the keyword are excluded — the shape alone FPs on serials."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.DRUG_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("NDC")
                    .optional().whitespace_char()
                    .between(4, 5).digit()
                    .char("-")
                    .between(3, 4).digit()
                    .char("-")
                    .between(1, 2).digit()
                .end()
                .group()
                    .string("ATC")
                    .optional().whitespace_char()
                    .range("A", "Z")
                    .exactly(2).digit()
                    .exactly(2).range("A", "Z")
                    .exactly(2).digit()
                .end()
            .end()
            .word_boundary()
        )


class GenomicAccessionBank(RegexBank):
    """UniProt, Ensembl, RefSeq, PDB accessions. PDB (1ABC) requires at least
    one letter so plain 4-digit numbers/years stay out; ordered after
    ValueWithUnitBank so 16GB is not read as a PDB id."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.GENOMIC_ACCESSION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                # Ensembl: ENS[GTPE]\d{11} (optionally species-prefixed)
                .group()
                    .string("ENS")
                    .between(0, 4).range("A", "Z")
                    .any_of_chars("GTPE")
                    .exactly(11).digit()
                .end()
                # RefSeq: NM_000546
                .group()
                    .any_of_chars("NX")
                    .any_of_chars("MRPGC")
                    .char("_")
                    .between(6, 9).digit()
                    .optional().group().char(".").one_or_more().digit().end()
                .end()
                # UniProt: P12345
                .group()
                    .any_of_chars("OPQ")
                    .digit()
                    .exactly(3).subexpression(_UPPER_ALNUM)
                    .digit()
                .end()
                # PDB: 1ABC — lookahead requires a letter among the last 3
                .group()
                    .digit()
                    .assert_ahead()
                        .between(0, 2).digit()
                        .range("A", "Z")
                    .end()
                    .exactly(3).subexpression(_UPPER_ALNUM)
                .end()
            .end()
            .word_boundary()
        )


class HGVSVariantBank(RegexBank):
    """HGVS variant notation: c.76A>T, p.Lys76Asn."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HGVS_VARIANT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                # coding/genomic: c.76A>T, g.123_124del
                .group()
                    .any_of_chars("cgmn")
                    .char(".")
                    .one_or_more().digit()
                    .optional().group().char("_").one_or_more().digit().end()
                    .any_of()
                        .group()
                            .any_of_chars("ACGT")
                            .char(">")
                            .any_of_chars("ACGT")
                        .end()
                        .group()
                            .any_of().string("del").string("ins").string("dup").end()
                            .zero_or_more().any_of_chars("ACGT")
                        .end()
                    .end()
                .end()
                # protein: p.Lys76Asn
                .group()
                    .string("p.")
                    .range("A", "Z")
                    .exactly(2).range("a", "z")
                    .one_or_more().digit()
                    .range("A", "Z")
                    .exactly(2).range("a", "z")
                .end()
            .end()
            .word_boundary()
        )


class ChemicalIdBank(RegexBank):
    """InChIKey, keyword-gated PubChem CID, ChEMBL ids. SMILES excluded —
    it is a grammar, not a token format (see StructuralIdentifier.CHEMICAL_ID)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CHEMICAL_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .exactly(14).range("A", "Z")
                    .char("-")
                    .exactly(10).range("A", "Z")
                    .char("-")
                    .range("A", "Z")
                .end()
                .group()
                    .string("CID")
                    .optional().whitespace_char()
                    .between(1, 9).digit()
                .end()
                .group()
                    .string("CHEMBL")
                    .between(1, 7).digit()
                .end()
            .end()
            .word_boundary()
        )


class ClinicalTrialIdBank(RegexBank):
    """NCT trial ids, PMIDs, ORCID ids."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CLINICAL_TRIAL_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("NCT").exactly(8).digit().end()
                .group()
                    .string("PMID")
                    .optional().char(":")
                    .optional().whitespace_char()
                    .between(6, 9).digit()
                .end()
                .group()
                    .exactly(4).digit().char("-")
                    .exactly(4).digit().char("-")
                    .exactly(4).digit().char("-")
                    .exactly(3).digit()
                    .any_of_chars("0123456789X")
                .end()
            .end()
            .word_boundary()
        )


class HealthcareProviderIdBank(RegexBank):
    """Keyword-gated NPI and DEA provider identifiers."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HEALTHCARE_PROVIDER_ID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.MEDICAL

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("NPI")
                    .optional().char(":")
                    .optional().whitespace_char()
                    .exactly(10).digit()
                .end()
                .group()
                    .string("DEA")
                    .optional().char(":")
                    .optional().whitespace_char()
                    .exactly(2).range("A", "Z")
                    .exactly(7).digit()
                .end()
            .end()
            .word_boundary()
        )

# ============================================================================
# identity/case_id.jl — CaseID: official-case identity/versioning key.
# ============================================================================

"""
    CaseID

Immutable identity/versioning key for an official benchmark case.

# Fields
- `family::Symbol`         : e.g. `:SimplePursuit`
- `variant::Symbol`        : e.g. `:Nominal`
- `version::VersionNumber` : e.g. `v"1"`
"""
struct CaseID
    family::Symbol
    variant::Symbol
    version::VersionNumber
end

Base.string(id::CaseID) = "$(id.family).$(id.variant)_v$(id.version.major)"
Base.show(io::IO, id::CaseID) = print(io, string(id))
Base.:(==)(a::CaseID, b::CaseID) = a.family == b.family && a.variant == b.variant && a.version == b.version
Base.hash(id::CaseID, h::UInt) = hash((id.family, id.variant, id.version), h)

def assert_dispatch_integrity(declared_set: set, dispatched_set: set, context_name: str) -> None:
    """
    Ensure that a set of declared capabilities matches the set actually dispatched in code.
    Prevents silent fallbacks or false capability reports.

    Args:
        declared_set: The set of keys declared to be supported (e.g., in a capabilities dict).
        dispatched_set: The set of keys actually handled by dispatch branches in the code.
        context_name: A descriptive string for error messages (e.g., "_ELEMENT_LARGE_DEF").
    """
    missing_branch = sorted(declared_set - dispatched_set)
    missing_decl = sorted(dispatched_set - declared_set)
    if missing_branch:
        raise AssertionError(
            f"[{context_name}] declares {missing_branch} but no assembly/dispatch "
            f"branch handles them. The capabilities report would be lying."
        )
    if missing_decl:
        raise AssertionError(
            f"[{context_name}]: {missing_decl} are dispatched in code but not declared "
            f"in the capabilities map. They might fail closed silently. Declare them."
        )

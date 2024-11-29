std::optional<AddressPosition> AddrManImpl::FindAddressEntry(const CAddress& addr)
{
    LOCK(cs);
    Check();
    auto entry = FindAddressEntry_(addr);
    Check();
    return entry;
}
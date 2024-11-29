std::optional<AddressPosition> AddrMan::FindAddressEntry(const CAddress& addr)
{
    return m_impl->FindAddressEntry(addr);
}
std::vector<CAddress> AddrManImpl::GetAddr(size_t max_addresses, size_t max_pct, std::optional<Network> network, const bool filtered) const
{
    LOCK(cs);
    Check();
    auto addresses = GetAddr_(max_addresses, max_pct, network, filtered);
    Check();
    return addresses;
}
std::vector<CAddress> AddrMan::GetAddr(size_t max_addresses, size_t max_pct, std::optional<Network> network, const bool filtered) const
{
    return m_impl->GetAddr(max_addresses, max_pct, network, filtered);
}
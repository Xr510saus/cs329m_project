size_t AddrMan::Size(std::optional<Network> net, std::optional<bool> in_new) const
{
    return m_impl->Size(net, in_new);
}
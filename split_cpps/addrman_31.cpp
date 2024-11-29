size_t AddrManImpl::Size(std::optional<Network> net, std::optional<bool> in_new) const
{
    LOCK(cs);
    Check();
    auto ret = Size_(net, in_new);
    Check();
    return ret;
}
std::vector<std::pair<AddrInfo, AddressPosition>> AddrManImpl::GetEntries(bool from_tried) const
{
    LOCK(cs);
    Check();
    auto addrInfos = GetEntries_(from_tried);
    Check();
    return addrInfos;
}
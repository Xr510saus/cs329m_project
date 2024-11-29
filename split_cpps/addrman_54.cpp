std::vector<std::pair<AddrInfo, AddressPosition>> AddrMan::GetEntries(bool use_tried) const
{
    return m_impl->GetEntries(use_tried);
}
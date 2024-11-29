size_t AddrManImpl::Size_(std::optional<Network> net, std::optional<bool> in_new) const
{
    AssertLockHeld(cs);

    if (!net.has_value()) {
        if (in_new.has_value()) {
            return *in_new ? nNew : nTried;
        } else {
            return vRandom.size();
        }
    }
    if (auto it = m_network_counts.find(*net); it != m_network_counts.end()) {
        auto net_count = it->second;
        if (in_new.has_value()) {
            return *in_new ? net_count.n_new : net_count.n_tried;
        } else {
            return net_count.n_new + net_count.n_tried;
        }
    }
    return 0;
}
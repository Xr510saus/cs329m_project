std::vector<CAddress> AddrManImpl::GetAddr_(size_t max_addresses, size_t max_pct, std::optional<Network> network, const bool filtered) const
{
    AssertLockHeld(cs);

    size_t nNodes = vRandom.size();
    if (max_pct != 0) {
        nNodes = max_pct * nNodes / 100;
    }
    if (max_addresses != 0) {
        nNodes = std::min(nNodes, max_addresses);
    }

    // gather a list of random nodes, skipping those of low quality
    const auto now{Now<NodeSeconds>()};
    std::vector<CAddress> addresses;
    addresses.reserve(nNodes);
    for (unsigned int n = 0; n < vRandom.size(); n++) {
        if (addresses.size() >= nNodes)
            break;

        int nRndPos = insecure_rand.randrange(vRandom.size() - n) + n;
        SwapRandom(n, nRndPos);
        const auto it{mapInfo.find(vRandom[n])};
        assert(it != mapInfo.end());

        const AddrInfo& ai{it->second};

        // Filter by network (optional)
        if (network != std::nullopt && ai.GetNetClass() != network) continue;

        // Filter for quality
        if (ai.IsTerrible(now) && filtered) continue;

        addresses.push_back(ai);
    }
    LogDebug(BCLog::ADDRMAN, "GetAddr returned %d random addresses\n", addresses.size());
    return addresses;
}
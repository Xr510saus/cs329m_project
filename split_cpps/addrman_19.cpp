std::pair<CAddress, NodeSeconds> AddrManImpl::Select_(bool new_only, const std::unordered_set<Network>& networks) const
{
    AssertLockHeld(cs);

    if (vRandom.empty()) return {};

    size_t new_count = nNew;
    size_t tried_count = nTried;

    if (!networks.empty()) {
        new_count = 0;
        tried_count = 0;
        for (auto& network : networks) {
            auto it = m_network_counts.find(network);
            if (it == m_network_counts.end()) {
                continue;
            }
            auto counts = it->second;
            new_count += counts.n_new;
            tried_count += counts.n_tried;
        }
    }

    if (new_only && new_count == 0) return {};
    if (new_count + tried_count == 0) return {};

    // Decide if we are going to search the new or tried table
    // If either option is viable, use a 50% chance to choose
    bool search_tried;
    if (new_only || tried_count == 0) {
        search_tried = false;
    } else if (new_count == 0) {
        search_tried = true;
    } else {
        search_tried = insecure_rand.randbool();
    }

    const int bucket_count{search_tried ? ADDRMAN_TRIED_BUCKET_COUNT : ADDRMAN_NEW_BUCKET_COUNT};

    // Loop through the addrman table until we find an appropriate entry
    double chance_factor = 1.0;
    while (1) {
        // Pick a bucket, and an initial position in that bucket.
        int bucket = insecure_rand.randrange(bucket_count);
        int initial_position = insecure_rand.randrange(ADDRMAN_BUCKET_SIZE);

        // Iterate over the positions of that bucket, starting at the initial one,
        // and looping around.
        int i, position;
        nid_type node_id;
        for (i = 0; i < ADDRMAN_BUCKET_SIZE; ++i) {
            position = (initial_position + i) % ADDRMAN_BUCKET_SIZE;
            node_id = GetEntry(search_tried, bucket, position);
            if (node_id != -1) {
                if (!networks.empty()) {
                    const auto it{mapInfo.find(node_id)};
                    if (Assume(it != mapInfo.end()) && networks.contains(it->second.GetNetwork())) break;
                } else {
                    break;
                }
            }
        }

        // If the bucket is entirely empty, start over with a (likely) different one.
        if (i == ADDRMAN_BUCKET_SIZE) continue;

        // Find the entry to return.
        const auto it_found{mapInfo.find(node_id)};
        assert(it_found != mapInfo.end());
        const AddrInfo& info{it_found->second};

        // With probability GetChance() * chance_factor, return the entry.
        if (insecure_rand.randbits<30>() < chance_factor * info.GetChance() * (1 << 30)) {
            LogDebug(BCLog::ADDRMAN, "Selected %s from %s\n", info.ToStringAddrPort(), search_tried ? "tried" : "new");
            return {info, info.m_last_try};
        }

        // Otherwise start over with a (likely) different bucket, and increased chance factor.
        chance_factor *= 1.2;
    }
}
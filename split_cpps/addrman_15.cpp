bool AddrManImpl::AddSingle(const CAddress& addr, const CNetAddr& source, std::chrono::seconds time_penalty)
{
    AssertLockHeld(cs);

    if (!addr.IsRoutable())
        return false;

    nid_type nId;
    AddrInfo* pinfo = Find(addr, &nId);

    // Do not set a penalty for a source's self-announcement
    if (addr == source) {
        time_penalty = 0s;
    }

    if (pinfo) {
        // periodically update nTime
        const bool currently_online{NodeClock::now() - addr.nTime < 24h};
        const auto update_interval{currently_online ? 1h : 24h};
        if (pinfo->nTime < addr.nTime - update_interval - time_penalty) {
            pinfo->nTime = std::max(NodeSeconds{0s}, addr.nTime - time_penalty);
        }

        // add services
        pinfo->nServices = ServiceFlags(pinfo->nServices | addr.nServices);

        // do not update if no new information is present
        if (addr.nTime <= pinfo->nTime) {
            return false;
        }

        // do not update if the entry was already in the "tried" table
        if (pinfo->fInTried)
            return false;

        // do not update if the max reference count is reached
        if (pinfo->nRefCount == ADDRMAN_NEW_BUCKETS_PER_ADDRESS)
            return false;

        // stochastic test: previous nRefCount == N: 2^N times harder to increase it
        if (pinfo->nRefCount > 0) {
            const int nFactor{1 << pinfo->nRefCount};
            if (insecure_rand.randrange(nFactor) != 0) return false;
        }
    } else {
        pinfo = Create(addr, source, &nId);
        pinfo->nTime = std::max(NodeSeconds{0s}, pinfo->nTime - time_penalty);
    }

    int nUBucket = pinfo->GetNewBucket(nKey, source, m_netgroupman);
    int nUBucketPos = pinfo->GetBucketPosition(nKey, true, nUBucket);
    bool fInsert = vvNew[nUBucket][nUBucketPos] == -1;
    if (vvNew[nUBucket][nUBucketPos] != nId) {
        if (!fInsert) {
            AddrInfo& infoExisting = mapInfo[vvNew[nUBucket][nUBucketPos]];
            if (infoExisting.IsTerrible() || (infoExisting.nRefCount > 1 && pinfo->nRefCount == 0)) {
                // Overwrite the existing new table entry.
                fInsert = true;
            }
        }
        if (fInsert) {
            ClearNew(nUBucket, nUBucketPos);
            pinfo->nRefCount++;
            vvNew[nUBucket][nUBucketPos] = nId;
            const auto mapped_as{m_netgroupman.GetMappedAS(addr)};
            LogDebug(BCLog::ADDRMAN, "Added %s%s to new[%i][%i]\n",
                     addr.ToStringAddrPort(), (mapped_as ? strprintf(" mapped to AS%i", mapped_as) : ""), nUBucket, nUBucketPos);
        } else {
            if (pinfo->nRefCount == 0) {
                Delete(nId);
            }
        }
    }
    return fInsert;
}
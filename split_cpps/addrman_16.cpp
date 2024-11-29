bool AddrManImpl::Good_(const CService& addr, bool test_before_evict, NodeSeconds time)
{
    AssertLockHeld(cs);

    nid_type nId;

    m_last_good = time;

    AddrInfo* pinfo = Find(addr, &nId);

    // if not found, bail out
    if (!pinfo) return false;

    AddrInfo& info = *pinfo;

    // update info
    info.m_last_success = time;
    info.m_last_try = time;
    info.nAttempts = 0;
    // nTime is not updated here, to avoid leaking information about
    // currently-connected peers.

    // if it is already in the tried set, don't do anything else
    if (info.fInTried) return false;

    // if it is not in new, something bad happened
    if (!Assume(info.nRefCount > 0)) return false;


    // which tried bucket to move the entry to
    int tried_bucket = info.GetTriedBucket(nKey, m_netgroupman);
    int tried_bucket_pos = info.GetBucketPosition(nKey, false, tried_bucket);

    // Will moving this address into tried evict another entry?
    if (test_before_evict && (vvTried[tried_bucket][tried_bucket_pos] != -1)) {
        if (m_tried_collisions.size() < ADDRMAN_SET_TRIED_COLLISION_SIZE) {
            m_tried_collisions.insert(nId);
        }
        // Output the entry we'd be colliding with, for debugging purposes
        auto colliding_entry = mapInfo.find(vvTried[tried_bucket][tried_bucket_pos]);
        LogDebug(BCLog::ADDRMAN, "Collision with %s while attempting to move %s to tried table. Collisions=%d\n",
                 colliding_entry != mapInfo.end() ? colliding_entry->second.ToStringAddrPort() : "",
                 addr.ToStringAddrPort(),
                 m_tried_collisions.size());
        return false;
    } else {
        // move nId to the tried tables
        MakeTried(info, nId);
        const auto mapped_as{m_netgroupman.GetMappedAS(addr)};
        LogDebug(BCLog::ADDRMAN, "Moved %s%s to tried[%i][%i]\n",
                 addr.ToStringAddrPort(), (mapped_as ? strprintf(" mapped to AS%i", mapped_as) : ""), tried_bucket, tried_bucket_pos);
        return true;
    }
}
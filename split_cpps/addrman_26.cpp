std::pair<CAddress, NodeSeconds> AddrManImpl::SelectTriedCollision_()
{
    AssertLockHeld(cs);

    if (m_tried_collisions.size() == 0) return {};

    std::set<nid_type>::iterator it = m_tried_collisions.begin();

    // Selects a random element from m_tried_collisions
    std::advance(it, insecure_rand.randrange(m_tried_collisions.size()));
    nid_type id_new = *it;

    // If id_new not found in mapInfo remove it from m_tried_collisions
    if (mapInfo.count(id_new) != 1) {
        m_tried_collisions.erase(it);
        return {};
    }

    const AddrInfo& newInfo = mapInfo[id_new];

    // which tried bucket to move the entry to
    int tried_bucket = newInfo.GetTriedBucket(nKey, m_netgroupman);
    int tried_bucket_pos = newInfo.GetBucketPosition(nKey, false, tried_bucket);

    const AddrInfo& info_old = mapInfo[vvTried[tried_bucket][tried_bucket_pos]];
    return {info_old, info_old.m_last_try};
}
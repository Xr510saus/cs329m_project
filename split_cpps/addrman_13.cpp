void AddrManImpl::ClearNew(int nUBucket, int nUBucketPos)
{
    AssertLockHeld(cs);

    // if there is an entry in the specified bucket, delete it.
    if (vvNew[nUBucket][nUBucketPos] != -1) {
        nid_type nIdDelete = vvNew[nUBucket][nUBucketPos];
        AddrInfo& infoDelete = mapInfo[nIdDelete];
        assert(infoDelete.nRefCount > 0);
        infoDelete.nRefCount--;
        vvNew[nUBucket][nUBucketPos] = -1;
        LogDebug(BCLog::ADDRMAN, "Removed %s from new[%i][%i]\n", infoDelete.ToStringAddrPort(), nUBucket, nUBucketPos);
        if (infoDelete.nRefCount == 0) {
            Delete(nIdDelete);
        }
    }
}
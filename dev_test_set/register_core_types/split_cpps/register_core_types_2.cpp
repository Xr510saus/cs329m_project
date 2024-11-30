void register_core_settings() {
	// Since in register core types, globals may not be present.
	GLOBAL_DEF(PropertyInfo(Variant::INT, "network/limits/tcp/connect_timeout_seconds", PROPERTY_HINT_RANGE, "1,1800,1"), (30));
	GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "network/limits/packet_peer_stream/max_buffer_po2", PROPERTY_HINT_RANGE, "0,64,1,or_greater"), (16));
	GLOBAL_DEF(PropertyInfo(Variant::STRING, "network/tls/certificate_bundle_override", PROPERTY_HINT_FILE, "*.crt"), "");

	GLOBAL_DEF("threading/worker_pool/max_threads", -1);
	GLOBAL_DEF("threading/worker_pool/low_priority_thread_ratio", 0.3);
}
void register_core_types() {
	OS::get_singleton()->benchmark_begin_measure("Core", "Register Types");

	//consistency check
	static_assert(sizeof(Callable) <= 16);

	ObjectDB::setup();

	StringName::setup();
	_time = memnew(Time);
	ResourceLoader::initialize();

	register_global_constants();

	Variant::register_types();

	CoreStringNames::create();

	resource_format_po.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_po);

	resource_saver_binary.instantiate();
	ResourceSaver::add_resource_format_saver(resource_saver_binary);
	resource_loader_binary.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_binary);

	resource_format_importer.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_importer);

	resource_format_importer_saver.instantiate();
	ResourceSaver::add_resource_format_saver(resource_format_importer_saver);

	resource_format_image.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_image);

	GDREGISTER_CLASS(Object);

	GDREGISTER_ABSTRACT_CLASS(Script);
	GDREGISTER_ABSTRACT_CLASS(ScriptLanguage);

	GDREGISTER_VIRTUAL_CLASS(ScriptExtension);
	GDREGISTER_VIRTUAL_CLASS(ScriptLanguageExtension);

	GDREGISTER_CLASS(RefCounted);
	GDREGISTER_CLASS(WeakRef);
	GDREGISTER_CLASS(Resource);
	GDREGISTER_VIRTUAL_CLASS(MissingResource);
	GDREGISTER_CLASS(Image);

	GDREGISTER_CLASS(Shortcut);
	GDREGISTER_ABSTRACT_CLASS(InputEvent);
	GDREGISTER_ABSTRACT_CLASS(InputEventWithModifiers);
	GDREGISTER_ABSTRACT_CLASS(InputEventFromWindow);
	GDREGISTER_CLASS(InputEventKey);
	GDREGISTER_CLASS(InputEventShortcut);
	GDREGISTER_ABSTRACT_CLASS(InputEventMouse);
	GDREGISTER_CLASS(InputEventMouseButton);
	GDREGISTER_CLASS(InputEventMouseMotion);
	GDREGISTER_CLASS(InputEventJoypadButton);
	GDREGISTER_CLASS(InputEventJoypadMotion);
	GDREGISTER_CLASS(InputEventScreenDrag);
	GDREGISTER_CLASS(InputEventScreenTouch);
	GDREGISTER_CLASS(InputEventAction);
	GDREGISTER_ABSTRACT_CLASS(InputEventGesture);
	GDREGISTER_CLASS(InputEventMagnifyGesture);
	GDREGISTER_CLASS(InputEventPanGesture);
	GDREGISTER_CLASS(InputEventMIDI);

	// Network
	GDREGISTER_ABSTRACT_CLASS(IP);

	GDREGISTER_ABSTRACT_CLASS(StreamPeer);
	GDREGISTER_CLASS(StreamPeerExtension);
	GDREGISTER_CLASS(StreamPeerBuffer);
	GDREGISTER_CLASS(StreamPeerGZIP);
	GDREGISTER_CLASS(StreamPeerTCP);
	GDREGISTER_CLASS(TCPServer);

	GDREGISTER_ABSTRACT_CLASS(PacketPeer);
	GDREGISTER_CLASS(PacketPeerExtension);
	GDREGISTER_CLASS(PacketPeerStream);
	GDREGISTER_CLASS(PacketPeerUDP);
	GDREGISTER_CLASS(UDPServer);

	GDREGISTER_ABSTRACT_CLASS(WorkerThreadPool);

	ClassDB::register_custom_instance_class<HTTPClient>();

	// Crypto
	GDREGISTER_CLASS(HashingContext);
	GDREGISTER_CLASS(AESContext);
	ClassDB::register_custom_instance_class<X509Certificate>();
	ClassDB::register_custom_instance_class<CryptoKey>();
	GDREGISTER_ABSTRACT_CLASS(TLSOptions);
	ClassDB::register_custom_instance_class<HMACContext>();
	ClassDB::register_custom_instance_class<Crypto>();
	ClassDB::register_custom_instance_class<StreamPeerTLS>();
	ClassDB::register_custom_instance_class<PacketPeerDTLS>();
	ClassDB::register_custom_instance_class<DTLSServer>();

	resource_format_saver_crypto.instantiate();
	ResourceSaver::add_resource_format_saver(resource_format_saver_crypto);
	resource_format_loader_crypto.instantiate();
	ResourceLoader::add_resource_format_loader(resource_format_loader_crypto);

	resource_loader_json.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_json);

	resource_saver_json.instantiate();
	ResourceSaver::add_resource_format_saver(resource_saver_json);

	GDREGISTER_CLASS(MainLoop);
	GDREGISTER_CLASS(Translation);
	GDREGISTER_CLASS(TranslationDomain);
	GDREGISTER_CLASS(OptimizedTranslation);
	GDREGISTER_CLASS(UndoRedo);
	GDREGISTER_CLASS(TriangleMesh);

	GDREGISTER_CLASS(ResourceFormatLoader);
	GDREGISTER_CLASS(ResourceFormatSaver);

	GDREGISTER_ABSTRACT_CLASS(FileAccess);
	GDREGISTER_ABSTRACT_CLASS(DirAccess);
	GDREGISTER_CLASS(core_bind::Thread);
	GDREGISTER_CLASS(core_bind::Mutex);
	GDREGISTER_CLASS(core_bind::Semaphore);

	GDREGISTER_CLASS(XMLParser);
	GDREGISTER_CLASS(JSON);

	GDREGISTER_CLASS(ConfigFile);

	GDREGISTER_CLASS(PCKPacker);

	GDREGISTER_CLASS(PackedDataContainer);
	GDREGISTER_ABSTRACT_CLASS(PackedDataContainerRef);
	GDREGISTER_CLASS(AStar3D);
	GDREGISTER_CLASS(AStar2D);
	GDREGISTER_CLASS(AStarGrid2D);
	GDREGISTER_CLASS(EncodedObjectAsID);
	GDREGISTER_CLASS(RandomNumberGenerator);

	GDREGISTER_ABSTRACT_CLASS(ImageFormatLoader);
	GDREGISTER_CLASS(ImageFormatLoaderExtension);
	GDREGISTER_ABSTRACT_CLASS(ResourceImporter);

	GDREGISTER_CLASS(GDExtension);

	GDREGISTER_ABSTRACT_CLASS(GDExtensionManager);

	GDREGISTER_ABSTRACT_CLASS(ResourceUID);

	GDREGISTER_CLASS(EngineProfiler);

	resource_uid = memnew(ResourceUID);

	gdextension_manager = memnew(GDExtensionManager);

	resource_loader_gdextension.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_gdextension);

	ip = IP::create();

	_geometry_2d = memnew(core_bind::Geometry2D);
	_geometry_3d = memnew(core_bind::Geometry3D);

	_resource_loader = memnew(core_bind::ResourceLoader);
	_resource_saver = memnew(core_bind::ResourceSaver);
	_os = memnew(core_bind::OS);
	_engine = memnew(core_bind::Engine);
	_classdb = memnew(core_bind::special::ClassDB);
	_marshalls = memnew(core_bind::Marshalls);
	_engine_debugger = memnew(core_bind::EngineDebugger);

	GDREGISTER_NATIVE_STRUCT(ObjectID, "uint64_t id = 0");
	GDREGISTER_NATIVE_STRUCT(AudioFrame, "float left;float right");
	GDREGISTER_NATIVE_STRUCT(ScriptLanguageExtensionProfilingInfo, "StringName signature;uint64_t call_count;uint64_t total_time;uint64_t self_time");

	worker_thread_pool = memnew(WorkerThreadPool);

	OS::get_singleton()->benchmark_end_measure("Core", "Register Types");
}
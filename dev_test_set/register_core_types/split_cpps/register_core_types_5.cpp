void unregister_core_extensions() {
	OS::get_singleton()->benchmark_begin_measure("Core", "Unregister Extensions");

	if (_is_core_extensions_registered) {
		gdextension_manager->deinitialize_extensions(GDExtension::INITIALIZATION_LEVEL_CORE);
	}
	GDExtension::finalize_gdextensions();

	OS::get_singleton()->benchmark_end_measure("Core", "Unregister Extensions");
}
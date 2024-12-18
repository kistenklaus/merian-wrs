#include "./renderdoc.hpp"
#include "renderdoc_app.h"

static RENDERDOC_API_1_1_2* rdoc_api = nullptr;

void renderdoc::init() {
    if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
        assert(ret == 1);
    }
}

void renderdoc::startCapture() {
    if (rdoc_api)
        rdoc_api->StartFrameCapture(nullptr, nullptr);
}

void renderdoc::stopCapture() {
  if(rdoc_api) rdoc_api->EndFrameCapture(nullptr, nullptr);
}

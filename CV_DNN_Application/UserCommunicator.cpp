#pragma unmanaged
#include "UserCommunicator.h"
#include <stdlib.h>
#include <windows.h>

namespace OpenCVApp {
	
	std::string UserCommunicator::askForFilename() {
		OPENFILENAMEA ofn;
		char nameBuff[MAX_PATH] = "";
		
		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.lpstrFilter = "‰æ‘œƒtƒ@ƒCƒ‹(*.jpg,*.jpeg,,*.bmp,*.png)\0*.jpg;*.bmp;*.png;*.jpeg\0";
		ofn.lpstrFile = nameBuff;
		ofn.nMaxFile = MAX_PATH;
		ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetOpenFileNameA(&ofn)) {
			return std::string(nameBuff);
		}
		else {
			return "";
		}
	}
	
}
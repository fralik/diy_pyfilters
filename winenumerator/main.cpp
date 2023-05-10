// https://docs.microsoft.com/en-us/windows/win32/directshow/selecting-a-capture-device

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Windows.h>
#include <cmath>
#include <dshow.h>
#include <comutil.h>
#include <string>
#include <vector>
#include <format>


// be sure to link the lib with class identifiers (CLSIDs) and interface identifiers (IIDs).
#pragma comment(lib, "strmiids")
// string conversion library
#pragma comment(lib, "comsuppwd.lib")

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


HRESULT EnumerateDevices(REFGUID category, IEnumMoniker** ppEnum)
{
    // Create the System Device Enumerator.
    ICreateDevEnum* pDevEnum;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
        CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pDevEnum));

    if (SUCCEEDED(hr))
    {
        // Create an enumerator for the category.
        hr = pDevEnum->CreateClassEnumerator(category, ppEnum, 0);
        if (hr == S_FALSE)
        {
            hr = VFW_E_NOT_FOUND;  // The category is empty. Treat as an error.
        }
        pDevEnum->Release();
    }
    return hr;
}

//const double e = 2.7182818284590452353602874713527;
//
//double sinh_impl(double x) {
//    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
//}
//
//double cosh_impl(double x) {
//    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
//}
//
//double tanh_impl(double x) {
//    return sinh_impl(x) / cosh_impl(x);
//}

namespace py = pybind11;

// https://docs.microsoft.com/en-us/windows/win32/directshow/deletemediatype
void _FreeMediaType(AM_MEDIA_TYPE& mt)
{
    if (mt.cbFormat != 0)
    {
        CoTaskMemFree((PVOID)mt.pbFormat);
        mt.cbFormat = 0;
        mt.pbFormat = NULL;
    }
    if (mt.pUnk != NULL)
    {
        // pUnk should not be used.
        mt.pUnk->Release();
        mt.pUnk = NULL;
    }
}

void _DeleteMediaType(AM_MEDIA_TYPE* pmt)
{
    if (pmt != NULL)
    {
        _FreeMediaType(*pmt);
        CoTaskMemFree(pmt);
    }
}

std::vector<std::tuple<int, std::string>> fetch_camera_names(IEnumMoniker* pEnum)
{
    std::vector<std::tuple<int, std::string>> res;
    IMoniker* pMoniker = NULL;
    int cameraId = 0;
    while (pEnum->Next(1, &pMoniker, NULL) == S_OK)
    {
        IPropertyBag* pPropBag;
        HRESULT hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag));
        if (FAILED(hr))
        {
            pMoniker->Release();
            continue;
        }

        // Get supported resolution
        // https://docs.microsoft.com/en-us/windows/win32/directshow/enumerating-media-types
        // https://stackoverflow.com/questions/4359775/windows-how-to-get-cameras-supported-resolutions/4360002#4360002
        IEnumPins* pEnum = NULL;
        IBaseFilter* pFilter = NULL;
        hr = pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&pFilter);
        if (FAILED(hr))
        {
            pMoniker->Release();
            continue;
        }

        hr = pFilter->EnumPins(&pEnum);
        if (FAILED(hr))
        {
            pFilter->Release();
            continue;
        }

        // Below we can fetch resolutions, but we omit this for now
        /*IPin* pPin = NULL;
        std::vector<std::tuple<int, int>> resolutions;
        while (S_OK == pEnum->Next(1, &pPin, NULL))
        {
            IEnumMediaTypes* pEnumMediaTypes = NULL;
            AM_MEDIA_TYPE* mediaType = NULL;
            VIDEOINFOHEADER* videoInfoHeader = NULL;
            HRESULT hr = pPin->EnumMediaTypes(&pEnumMediaTypes);
            if (FAILED(hr))
            {
                continue;
            }

            while (hr = pEnumMediaTypes->Next(1, &mediaType, NULL), hr == S_OK)
            {
                if ((mediaType->formattype == FORMAT_VideoInfo) &&
                    (mediaType->cbFormat >= sizeof(VIDEOINFOHEADER)) &&
                    (mediaType->pbFormat != NULL))
                {
                    videoInfoHeader = (VIDEOINFOHEADER*)mediaType->pbFormat;
                    videoInfoHeader->bmiHeader.biWidth;
                    videoInfoHeader->bmiHeader.biHeight;
                    
                    resolutions.push_back(std::tuple<int, int>{videoInfoHeader->bmiHeader.biWidth, videoInfoHeader->bmiHeader.biHeight});
                }
                _DeleteMediaType(mediaType);
            }
            pEnumMediaTypes->Release();
        }*/

        VARIANT var;
        VariantInit(&var);

        // Get description or friendly name.
        hr = pPropBag->Read(L"Description", &var, 0);
        if (FAILED(hr))
        {
            hr = pPropBag->Read(L"FriendlyName", &var, 0);
        }
        if (SUCCEEDED(hr))
        {
            auto stringName = _com_util::ConvertBSTRToString(var.bstrVal);
            
            // Here is how we may incorporate resolutions:
            //for (auto& resolution : resolutions) {
            //    res.push_back(std::tuple<int, std::string, std::tuple<int, int>>{cameraId, stringName, resolution});
			//}

            res.push_back(std::tuple<int, std::string>{cameraId++, stringName});
            VariantClear(&var);
        }

        pPropBag->Release();
        pMoniker->Release();
    }

    return res;
}

std::vector<std::tuple<int, std::string>> list_cameras()
{
    // py::list res;
    std::vector<std::tuple<int, std::string>> res;
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr))
    {
        IEnumMoniker* pEnum;
        hr = EnumerateDevices(CLSID_VideoInputDeviceCategory, &pEnum);
        if (SUCCEEDED(hr))
        {
            res = fetch_camera_names(pEnum);
            pEnum->Release();
        }
        CoUninitialize();
    }
    return res;
}

PYBIND11_MODULE(winenumerator, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: winenumerator

        .. autosummary::
           :toctree: _generate

           list_cameras
    )pbdoc";

    m.def("list_cameras", &list_cameras, R"pbdoc(
        List video capture devices and their resolution.

        Some other explanation about the list_cameras function.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

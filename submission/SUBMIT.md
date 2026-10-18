# lablab.ai submission sheet (copy-paste in this order)

## 1. Basic information
| Field | Value |
|---|---|
| Project title | Maintenance Copilot: Catch Failures Early |
| Short description | AI maintenance copilot on AMD (Gemma-4-31B-IT): flags machine drift 39h before fixed alarms, explains the cause with cited manual sources and recommends safe actions. |
| Long description | Paste the whole of `submission/long_description.md` |
| Technology & category tags | AMD Token Factory, Gemma, LLM, RAG, Predictive Maintenance, Industrial AI, Python, Streamlit, Docker, OpenAI-compatible API |

## 2. Cover image and presentation
| Field | Value |
|---|---|
| Cover image | Upload `submission/cover.png` (1600×900) |
| Video presentation | Your YouTube (unlisted) or Loom link, recorded from `submission/pitch_script.md` |
| Slide presentation | Upload `submission/slides.pdf` (7 slides) |

## 3. App hosting and repository
| Field | Value |
|---|---|
| Public GitHub repository | https://github.com/Vaish54224/maintenance-copilot |
| Demo application platform | Streamlit Community Cloud |
| Application URL | Your `https://<name>.streamlit.app` link after deploying |

### Deploy (once)
1. Open https://share.streamlit.io/deploy?repository=Vaish54224/maintenance-copilot&branch=main&mainModule=app.py and sign in with GitHub.
2. **Advanced settings** → Python 3.12 → **Secrets**, paste (with your key):
   ```
   AMD_API_KEY = "<your key>"
   AMD_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
   AMD_MODEL = "Gemma-4-31B-IT"
   ```
3. **Deploy**, wait ~2 min, then click *AI diagnosis → Diagnose with AMD* and one chat chip on the public URL.

## Final check before Submit
- [ ] App URL opens in a private/incognito window and Diagnose works
- [ ] Video link plays for someone who is not logged in
- [ ] Slides PDF and cover uploaded
- [ ] Click **Submit**

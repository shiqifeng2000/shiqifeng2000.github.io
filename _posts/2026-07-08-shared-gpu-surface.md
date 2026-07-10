---
layout: post
title: How to design a multiple video rendering lib with synchonizing feature.
subtitle: A cuda based gpu interlop case with synchonization.
gh-repo: shiqifeng2000/VideoSynchonizationPipeline
gh-badge: [star, fork, follow]
tags: [rust, raw_sync]
comments: true
mathjax: true
author: shiqifeng
---

{: .box-success}
A playback synchonizing design for mutiple videos on nvidia powered PCs.

![Crepe](../assets/img/2026-07-08/sddefault.jpg)

**Context: I need to embed a player into Unity to decode 3+ videos, render them, and synchronize them seamlessly with audio. Each video represents one portion of a larger screen, so every rendered frame must align exactly.**


In this post, I'd like to discuss a player design that renders multiple MP4 videos. I know that some systems, such as macOS, provide APIs like `AVVideoComposition`. However, my goal isn't just to "render" video — I need a cross-platform solution that can share the decoded surface with the actual renderer, so that an upper-layer cross-platform player or application manipulate the surface I share. The videos are 2K to 4K, running at 30 to 60 fps.

### First, how should we design the decoding workflow?

To ensure smooth decoding of all videos, hardware-accelerated decoding is a must. That means the frames we get from the decoder reside in GPU memory. For rendering, we can use OpenGL or DirectX for display, so the display container should also live on the GPU. Decoding and rendering are two distinct phases and often involve different formats — for example, CUDA produces NV12, while OpenGL/DirectX expects RGB. Typically, people copy the decoded GPU data to CPU RAM, convert the format, and copy it back to the rendering phase (OpenGL surface, DirectX shader, or framebuffer). This round-trip happens because they need to change the format or use CPU-bound tools like OpenCV.

```
                    Video Playback Pipeline

                ┌───────────────────────────┐
Compressed ---->| HW Decoder                |
H264/H265/AV1   | (NVDEC / VideoToolbox /   |
                | DXVA / VAAPI / MediaCodec)|
                └─────────────┬─────────────┘
                              │
                              │ Decoded Frame
                              ▼
                     GPU Memory (NV12/YUV420)

                              │
                              │ GPU → CPU Copy
                              ▼
                    +----------------------+
                    |     System Memory    |
                    |   NV12/YUV420 Frame  |
                    +----------------------+
                              │
             ┌────────────────┴─────────────────┐
             │                                  │
             │ CPU Color Conversion             │
             │ CPU Image Processing             │
             │ OpenCV / FFmpeg Filters          │
             │ AI Preprocessing                 │
             └────────────────┬─────────────────┘
                              │
                              │ RGB/BGRA
                              ▼
                    +----------------------+
                    |     System Memory    |
                    |      RGB Frame       |
                    +----------------------+
                              │
                              │ CPU → GPU Copy
                              ▼
                    GPU Texture / Surface
                              │
                              ▼
                    OpenGL / DirectX / Metal
                              │
                              ▼
                         Display Window
```

But let's do the math. A 4K video at 60 fps means 3840×2160×4×60 = 1990 MB/s of RGBA data transfer. With 5 concurrent videos, that's 5 × 1990 = 9.9 GB/s. If we copy from GPU to RAM, then from RAM back to the surface, we're talking about 9.9 GB/s in each direction. This approach becomes ridiculous when the player stalls due to PCIe bus bandwidth limits or surface upload constraints.


**PCI Bus Copy Speeds**


| PCIe Version | Per-Lane Data Rate | Encoding | **x1 Bandwidth** | **x4 Bandwidth** | **x8 Bandwidth** | **x16 Bandwidth** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PCIe 1.0** | 2.5 GT/s | 8b/10b | 250 MB/s | 1.0 GB/s | 2.0 GB/s | 4.0 GB/s |
| **PCIe 2.0** | 5.0 GT/s | 8b/10b | 500 MB/s | 2.0 GB/s | 4.0 GB/s | **8.0 GB/s** |
| **PCIe 3.0** | 8.0 GT/s | 128b/130b | 985 MB/s | 3.94 GB/s | 7.88 GB/s | **15.75 GB/s** |
| **PCIe 4.0** | 16.0 GT/s | 128b/130b | 1.97 GB/s | 7.88 GB/s | 15.75 GB/s | **31.5 GB/s** |
| **PCIe 5.0** | 32.0 GT/s | 128b/130b | 3.94 GB/s | 15.75 GB/s | 31.5 GB/s | **63.0 GB/s** |
| **PCIe 6.0** | 64.0 GT/s | PAM4 + FEC | 7.56 GB/s | 30.25 GB/s | 60.5 GB/s | **121.0 GB/s** |
| **PCIe 7.0** (expected 2025) | 128.0 GT/s | PAM4 | 32 GB/s | 128 GB/s | 256 GB/s | **512 GB/s** |




**So I think the best approach is to copy decoded video data directly from GPU to the surface, or Zero-Copy in other words**

Something like:
```
Bitstream
    │
    ▼
NVDEC
    │
    ▼
NV12 Frame (pitched)
    │
    │  (shared or mapped)
    ▼
CUDA / Fragment Shader
    │
    │  YUV→RGB, scaling, denoise,
    │  crop, AI, tone mapping
    ▼
OpenGL Texture / Render Target
    │
    ▼
Swapchain
    │
    ▼
Display
```
This way, image processing stays entirely within the GPU. In the CUDA world, I need to introduce the [NPP](https://docs.nvidia.com/cuda/archive/9.1/npp/index.html) library for format conversion and other processing tasks. This is a must if we're targeting Unity3D, since the surface it provides is always in RGBA format.

NPP doesn't support RGBA directly — only RGB. That's fine; we can convert with `nppiNV12ToRGB_8u_P2C3R_Ctx`, then use `cudaMemcpy2D` for pitching, and finally insert a 1 as the alpha value for every fourth item.

### Now that we have the surface data prepared, how do we fill it into the surface?

CUDA provides convenient APIs for interop. With `cudaGraphicsGLRegisterImage` or `cudaGraphicsD3D11RegisterImage`, we can register an OpenGL/D3D surface as a **CUDA resource**. Then, using `cudaGraphicsMapResources` and `cudaGraphicsSubResourceGetMappedArray`, we can obtain the resource buffer pointer or array, meaning the surface is ready to accept frame data.


```
OpenGL / D3D Texture (Render Surface)
                 │
                 │ Register
                 │
                 ▼
      CUDA Graphics Resource
```

```
Compressed Stream
        │
        ▼
   HW Decoder
        │
        ▼
  NV12 Decode Frame
        │
        │ Map Resource
        ▼
   CUDA Kernel
 (NV12→RGB, Scale...)
        │
        ▼
 Registered GL/D3D Surface
        │
        ▼
   Render / Present
```

### Okay, now everything is connected — but only for a single video.

We can spin up multiple decoding threads, register multiple surfaces, and render them independently. Something like:

```
                 One-time Initialization

+-----------+   +-----------+   +-----------+
| GL Tex #1 |   | GL Tex #2 |   | GL Tex #3 |
+-----+-----+   +-----+-----+   +-----+-----+
      |                 |                 |
 Register          Register          Register
      |                 |                 |
      v                 v                 v
 CUDA Resource1   CUDA Resource2   CUDA Resource3


                 Per-frame (Independent)

Video 1              Video 2              Video 3
--------             --------             --------

Bitstream1           Bitstream2           Bitstream3
     |                    |                    |
HW Decoder1          HW Decoder2          HW Decoder3
     |                    |                    |
NV12 Surface1        NV12 Surface2        NV12 Surface3
     |                    |                    |
Map Resource1        Map Resource2        Map Resource3
     |                    |                    |
CUDA Kernel1         CUDA Kernel2         CUDA Kernel3
     |                    |                    |
GL Texture1          GL Texture2          GL Texture3
     |                    |                    |
  Present1             Present2             Present3
```

Well, This design certainly works — but we need **synchronization**.

A typical video player uses one of three synchronization strategies:
1. Video aligns to audio
2. Audio aligns to video
3. Both video and audio align to a common timeline

I chose the third option, since we have multiple video streams. Each video file contains the same audio stream, so I pick the first audio stream found and use [cpal](https://docs.rs/cpal/latest/cpal/) for playback. I'll skip the audio details (like resampling) since this post is primarily about video.

Before synchronization, the picture looks like this:

```
                 One-time Initialization

+-----------+   +-----------+   +-----------+
| GL Tex #1 |   | GL Tex #2 |   | GL Tex #3 |
+-----+-----+   +-----+-----+   +-----+-----+
      |                 |                 |
 Register          Register          Register
      |                 |                 |
      v                 v                 v
 CUDA Resource1   CUDA Resource2   CUDA Resource3


                 Per-frame (Independent thread)

Video 1             Video 2             Video 3             Audio
--------            --------            --------            -----

Bitstream1          Bitstream2          Bitstream3         Bitstream4
     |                   |                   |                  |
HW Decoder1         HW Decoder2         HW Decoder3        Audio Demux
     |                   |                   |                  |
NV12 Surface1       NV12 Surface2       NV12 Surface3      Audio Frames
     |                   |                   |                  |
Map Resource1       Map Resource2       Map Resource3        Decode
     |                   |                   |                  |
CUDA Kernel1        CUDA Kernel2        CUDA Kernel3       Audio Buffer
     |                   |                   |                  |
GL Texture1         GL Texture2         GL Texture3        CPAL Output
     |                   |                   |                  |
  Present1           Present2            Present3           Speakers
```

Next, you’re probably considering timeline or time shift work. But you know, we need to synchonize the video at the frame level, so here's the thing, we need to `GROUP` all video frames first, then align video group with the audio pcm.
The grouping is simply a data collecting work, all threads must provide a frame, if any thread is not fast enough, grouping thread should wait for it. 
But the waiting, should not block any of the decoding process, or there shall be stucking experiences. So we need to decouple the decoding and grouping work, when decoding thread produces frames, all frames must be sent to grouping thread and got cached. The grouping thread then, loops against some timeline pattern, picking up the right frame and got them copied to surface buffer.

The design now should be something like this:
```
                         Independent Decode Threads
================================================================================

Video 1            Video 2            Video 3                 Audio
--------           --------           --------                -----

Bitstream1         Bitstream2         Bitstream3            Bitstream4
     |                  |                  |                     |
HW Decoder1        HW Decoder2        HW Decoder3           Audio Decode
     |                  |                  |                     |
Frame Cache1       Frame Cache2       Frame Cache3          PCM Buffer
     |                  |                  |                     |
     +------------------+------------------+---------------------+
                        |                  |
                        | decoded frames   |  decoded pcm
                        |                  |
                        v                  v
                 +-------------------------------+
                 |        Grouping Thread        |
                 |-------------------------------|
                 | Wait until every cache has    |
                 | a frame for current timeline  |
                 | Align video frames with PCM   |
                 | Select Frame Group            |
                 +---------------+---------------+
                                 |
                  +--------------+--------------+
                  |              |              |
                  v              v              v
               Video 1        Video 2        Video 3
            Map Resource   Map Resource   Map Resource
                  |              |              |
            CUDA Kernel    CUDA Kernel    CUDA Kernel
                  |              |              |
             GL Texture     GL Texture     GL Texture
                  |              |              |
              Present1       Present2       Present3
                                ||
                                ||
                        CPAL Audio Output
```

Since we are using NVDEC to decode the videos, the speed is ultra fast, if there's no blocking machenism, the gpu memory will soon be consumed and elipsed. So we need some ringbuffer design to harness the decoding thread, block if necessary, meanwhile keeps feeding gpu frame data to the grouping thread.
We could use 2 [mpsc](https://doc.rust-lang.org/stable/std/sync/mpsc/index.html) channels, one to receive from decoding thread and get cached, with frame index(grouping), pts(video/audio synchonizing), stream hash(seeking) and necessary info included. Then send to the other side, the rendering side to unload the frame to surface, then get sent back again to decoding thread.

Now the caching version:
```
                         Independent Decode Threads
================================================================================

Video 1            Video 2            Video 3                 Audio
--------           --------           --------                -----

Bitstream1         Bitstream2         Bitstream3            Bitstream4
     |                  |                  |                     |
HW Decoder1        HW Decoder2        HW Decoder3           Audio Decode
     |                  |                  |                     |
Frame Cache1       Frame Cache2       Frame Cache3          PCM Buffer
     |                  |                  |                     |
     +------------------+------------------+---------------------+
                        |                  ▲
                        | decoded frames   | recycled
                        v                  |
                +--------------------------------+
                |         Frame Cache            |
                |      (Ring Buffer Pool)        |
                +--------------------------------+
                        |                  ▲
                        | Load             | recycled
                        v                  |
                 +-------------------------------+
                 |        Grouping Thread        |
                 |-------------------------------|
                 | Wait until every cache has    |
                 | a frame for current timeline  |
                 | Align video frames with PCM   |
                 | Select Frame Group            |
                 +---------------+---------------+
                                 |
                  +--------------+--------------+
                  |              |              |
                  v              v              v
               Video 1        Video 2        Video 3
            Map Resource   Map Resource   Map Resource
                  |              |              |
            CUDA Kernel    CUDA Kernel    CUDA Kernel
                  |              |              |
             GL Texture     GL Texture     GL Texture
                  |              |              |
              Present1       Present2       Present3
                                ||
                                ||
                        CPAL Audio Output
```

### finally, we could consider the timeline work.

The grouping thread is in a forever looping, it keeps peeking the ring buffer pool, if any video/audio, put them to the corresponding video/audio cache. Each looping calculate the `current pts`(Presentation Timestamp) and the current latest video/audio pts, if video/audio `stream pts` is later than `current pts`, do the rendering.

### Cuda pipeline cons you dont want to miss.

The CUDA API works in a pipelined, async‑fashion. When you call `cudaMemcpy`, it doesn't actually block — it just queues up the work and returns immediately. That's great for performance, but it can bite you when synchronizing frames. If you copy multiple frames to surfaces in quick succession, you might end up with mismatched indices because the actual copying order doesn't strictly follow your calls. To fix this, you need to explicitly insert a sync point, like `cudaStreamSynchronize`, to ensure everything's done before moving on.

### Opengl restriction you dont want to miss.

OpenGL has a well‑known restriction: the thread that registers a GL resource (like a texture) must be the same one that renders into it. You can't offload rendering to another thread. So in our design, both registration and rendering have to live in the layer that owns the GL context — in our case, that's Unity3D.

## Summary
We got a multiple video synchronizing player design and built based on opengl, which is os-independtent somehow, if on windows, we could use some d3d11 api, yes there's d3d11 cuda interloping api as well.
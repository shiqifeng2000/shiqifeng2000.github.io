---
layout: post
title: Mutiple video decode+suface synchonizing design 
subtitle: A cuda based heterogeneous playback design
gh-repo: shiqifeng2000/VideoSynchonizationPipeline
gh-badge: [star, fork, follow]
tags: [rust, raw_sync]
comments: true
mathjax: true
author: shiqifeng
---

{: .box-success}
A playback synchonizing design for mutiple videos on nvidia powered PCs.

**Context: I will need to embed a `player` to unity to help decode 3+ videos and have them rendered and synchonized seamlessly against audios**

In this post, I woud like to discuss a player design that render multiple videos(mp4). I knew in some system such as Mac, there's `AVVideoComposition` API or something alike. But my intention is not to just 'render' it, but got some solution to share a surface to the actual player. So that the upper layer, a cross-platform player or application could just use the surface I produced to render. The video could be a 2k ~ 4k video with the framerate 30 ~ 60.

### So, firstly, where to place the decoded frame

If we are to make sure all the videos are decoded smoothly, hw-accelerated decoding must be implemented, when rendering, we could use opengl or directx for displaying.

But here's the math, a 4k video with 60fps means 3840x2160x1.5x60=712MB/s of data transfering, if there's 5 videos transfering at the same time, the speed shall be 5x712 = 3.5GB/s. If we are to copy gpu data to ram then ram to surface, it means 3.5GB/s of data from gpu to cpu, then 3.5GB/s of data from cpu to gpu for surface rendering. That's rediculous, the player shall stucks with normal PCIe bus copying speed or surface uploading restrictions.

---

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


**So I guess maybe the best way here is to copy decoded video data in gpu directly to surface**

Something like:

```
                Surface Pool
        +-------------------------+
        | Surface0                |
        | Surface1                |
        | Surface2                |
        +-------------------------+

              acquire()
                  │
                  ▼
        +-----------------+
        |   FREE Surface  |
        +-----------------+
                  │
                  │ fill pixels
                  ▼
        +-----------------+
        |     FILLED      |
        +-----------------+
                  │
                  │ queue/present
                  ▼
        +-----------------+
        |   RENDERING     |
        +-----------------+
                  │
                  │ GPU/display finished
                  ▼
        +-----------------+
        |    RELEASED     |
        +-----------------+
                  │
                  │ recycle
                  └──────────────────────────────┐
                                                 │
                                                 ▼
                                           acquire again
```

### Now that the data flow is designed, we now need a synchonizing pattern design. 

To synchronize the frames, the surface we malloced must be in a `Grouped` format. Say, 5 videos, 5 surfaces in a group. They are registered together, filled together.



PCI总线的拷贝速度，和我们常说的PCIe（PCI Express）是两回事。传统PCI总线理论上的最高速度是 **133 MB/s** 或 **533 MB/s**（64位/66MHz版本），而目前主流的PCIe总线速度要快得多，从几百MB/s到几十GB/s不等。

我把两者的速度标准整理成了表格，方便你对比：

### ⏳ 传统并行PCI总线速度

传统的PCI总线是一种**并行总线**，它的速度主要由总线宽度和时钟频率决定。以下是常见的几种规格：

| PCI 总线规格 | 总线宽度 | 时钟频率 | **理论带宽 (拷贝速度)** |
| :--- | :--- | :--- | :--- |
| **PCI 2.3** | 32-bit | 33 MHz | **133 MB/s** |
| **PCI 2.3** | 32-bit | 66 MHz | **266 MB/s** |
| **PCI 64** | 64-bit | 33 MHz | **266 MB/s** |
| **PCI 64** | 64-bit | 66 MHz | **533 MB/s** |

So maybe the best way is to get everything(including ffi) into rust program, just like libs(so/dylibs), but things turns out the algorithm model is composed of many python codes, they are associated in a way of depending each other and disk file searching/dynamic importing. This topology makes it impossible to rewrite them the c++/rust way. 

### First Try
So in our first attempt(version 1), we just did the work in a pipeline way, using stdin to stream all my yuv frames to python program, then stdout to pipe out. The pipeline is not cheap to get to work, since python code depend on many dependencies that secretly print something. I will need to design some header(separator) just like mp4 box or h26x annex-b way. to parse the real yuv content. 

stdin
```rust
let mut idx = self.frames % infsdrs_len;
let mut data = vec![];
data.extend_from_slice(STDIO_SEP.as_bytes());
data.extend_from_slice(&(self.frames as u32).to_be_bytes());
self.prosess_avframe(self.dec_frame, &mut data)?;
```

stdout reading
```rust
let std_finder = memchr::memmem::Finder::new(&*STDIO_SEP);
let sep_len = STDIO_SEP.len();
loop {
  if let Ok(n_rst) = tokio_read_stdio!(&mut stdout, &mut buf, 200) {
      match n_rst {
          Ok(n) => {
              if n > 0 {
                  cache.extend_from_slice(&buf[0..n]);
                  if let Some(m) = std_finder.find(&cache) {
                      if cache.len() >= total_frame_size + m {
                      ...
```

And of course, some more pitch copying work for irregular-sized videos. 

The result is, yes, it works, at the cost of:
1. Python needed, the program is not portable or not that easy to port, the whole system is just very 'dirty' 
2. Pipeline memory is often huge, and there's few guarantee to prevent abnormal case such as clamping or breaker.
3. Exception handling or the system design is complicated, we need to design a very robust patten to cover all abnormals.

### Second Try
Now that we got a dirty video enhanceing program up and running. We now want to evolve it. The evolving must cover the cons above. 

1. Python issue. I tried many ways, initialy the miniforge, then uv tool. Luckily, it turns out the [pyinstaller](https://github.com/pyinstaller/pyinstaller) could build the algorithm model into executable, after some adjustment to turn the relative path to absolute path in the python codes. I made the executable functional. This is not ideal, but could save a lot of deployment complexity.

2. Pipeline issue. To avoid large memory issue, I did consider semaphore before, but i will need these 2 terms
    - it must be cross platform, so posix only way is not preferred
    - algorithms will need more than 1 worker sometimes, so if i got 3 algo worker, at lease 3 semaphore must be created, this introduce sema management complexity
As a result, I start to consider this crate [raw_sync](https://github.com/elast0ny/raw_sync-rs), to create a super fast, low memory cost shared memory mutex/rwlock flag.

3. If I could fix the 2 issues above, that means the exception handling will shrink only to atomic flag, shm file processing and etc. Using a timeout and some flag design to report abnormal algorithm worker is much simplier than previous pure python + weight searching + pipeline design.

This time, I turn to [shared memory](https://github.com/elast0ny/shared_memory-rs) and [raw_sync](https://github.com/elast0ny/raw_sync-rs). raw_sync already provides some mutex wrapping, on posix it is build upon pthread then on windows it uses winapi::um::synchapi. So a mutex is now created.

So when i create a file using shared_memory, this shem crate shall map the file content into my ram, just like
```
                                    Shared mmap() File
     ┌────────────────────────────────────────────────────────────────────────────┐
     │                                                                            │
     │  +----------------+----------------------------------------------------+   │
     │  | 0 ~ 7          | AtomicU8 Flag (8 bytes / cache-line aligned)       |   │
     │  +----------------+----------------------------------------------------+   │
     │  | 8 ~ xxx        | pthread_mutex_t (PTHREAD_PROCESS_SHARED)           |   │
     │  +----------------+----------------------------------------------------+   │
     │  | Input Buffer   |                                                    |   │
     │  | (Rust → Python)|   Raw video frame / metadata                       |   │
     │  |                |                                                    |   │
     │  +----------------+----------------------------------------------------+   │
     │  | Output Buffer  |                                                    |   │
     │  | (Python→ Rust) |   Detection / AI result / processed frame          |   │
     │  |                |                                                    |   │
     │  +----------------+----------------------------------------------------+   │
     │                                                                            │
     └────────────────────────────────────────────────────────────────────────────┘
              ▲                                                    ▲
              │                                                    │
              │                                                    │
      write input frame                                   write AI result
              │                                                    │
              │                                                    │
              │                                                    │
  ┌──────────────────────────┐                        ┌──────────────────────────┐
  │      Rust Process        │                        │   Pyo3/Python Process    │
  │──────────────────────────│                        │──────────────────────────│
  │ HW Decode                │                        │ AI / CV Algorithm        │
  │ HW Video Processing      │                        │ NumPy / Torch / OpenCV   │
  │ WebRTC / Encoder         │                        │ CPU/GPU Inference        │
  └──────────────────────────┘                        └──────────────────────────┘
```

When the work is done, and I review what I did. The atomic and pthread looks like duplicate functions.

- AtomicU8 Flag is a shared memory based flag, to enable python end to use it, I build a python wheel project in rust and installed a python sdk to help deal with the flag. Both end could use this flag, looks seems to more to be a **Process Safe** pattern. The intesting part is, if I use the flag as the mutex lock, it takes about 10ns for each loop, more than a quick responding `spin lock`

```rust
while flag.load(Ordering::Relaxed) != CODE_CONSUMED {
    std::hint::spin_loop();
    if write_start.elapsed().as_secs() > timeout {
        bail!("Host timeout waiting for consumer flag CODE_CONSUMED")
    }
}
```


- If I use mutex to synchonize the buffer, sys-call shall be introduced. In posix, it is `pthread-mutex`, or maybe `futex`. Since it is kernel involing work, the user/kernel context swapping is some costing.

```rust
let lock = self.lock
  .try_lock(raw_sync::Timeout::Val(Duration::from_secs(timeout)))?;
```

It turns out the performance is no better than the atomicu8 flag, the only tradeoff for atomicu8 is, it takes up a CPU core when waiting.

## Summary
If you got much more resource such as cpu cores and memory, the best performance is atomicu8 shared memory. If cpu cores is limited but memory is sufficient, you may consider pipeline(stdin/out). If you need solid solution, with limited memory, you may need to consider the futex way.
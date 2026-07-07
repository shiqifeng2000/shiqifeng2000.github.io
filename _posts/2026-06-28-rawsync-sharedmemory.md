---
layout: post
title: Some thoughts about shared memory
subtitle: A raw_sync based shared memory
gh-repo: shiqifeng2000/VideoEnhancer
gh-badge: [star, fork, follow]
tags: [rust, raw_sync]
comments: true
mathjax: true
author: shiqifeng
---

{: .box-success}
This is a post to discuss how I implement the shared memory the rust way during the video enhancement scenario.

**Context: A video enchancement project to pipe video frames to python algorithm, get processed then back to encode/mux to new videos**

In this post, I would like to discuss the shared memory tech. This project is mainly about media processing - it includes FFmpeg, Libx26x, Zlib and Font libs, written purely in rust. But it is just one part of the glacier -- The algorithm to process image is the core. I need to work around the algorithm to enhance all video frames, only after that, can I encode and mux to new mp4/movs.

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
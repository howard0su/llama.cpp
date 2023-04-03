#pragma once

#if defined _MSC_VER || defined(__MINGW32__)

#if !defined(__MINGW32__)
#include <Windows.h>
#else
// ref: https://github.com/ggerganov/whisper.cpp/issues/168
#include <windows.h>
#endif

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

static void atomic_store(atomic_int *ptr, LONG val)
{
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int *ptr)
{
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int *ptr, LONG inc)
{
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_sub(atomic_int *ptr, LONG dec)
{
    return atomic_fetch_add(ptr, -(dec));
}

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t *out, void *unused, thread_ret_t (*func)(void *), void *arg)
{
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void *unused)
{
    return (int)WaitForSingleObject(thread, INFINITE);
}

static int sched_yield(void)
{
    Sleep(0);
    return 0;
}
#else
#include <pthread.h>
#include <stdatomic.h>

typedef void *thread_ret_t;
#endif
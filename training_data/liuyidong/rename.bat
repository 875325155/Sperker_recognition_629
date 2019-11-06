@echo off

set a=00

setlocal EnableDelayedExpansion

for %%n in (*.wav) do (

set /A a+=1

ren "%%n" "liuyidong!a!.wav"

)
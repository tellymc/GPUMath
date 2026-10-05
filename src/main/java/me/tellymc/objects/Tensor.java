package me.tellymc.objects;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class Tensor {

    private final CUdeviceptr pointer;
    private final int rows;
    private final int columns;
    private final int size;
    private final int sizeBytes;

    public Tensor(int rows, int columns, float[] input) {
        this.rows = rows;
        this.columns = columns;

        this.size = this.rows * this.columns;
        this.sizeBytes = this.size * Sizeof.FLOAT;
        this.pointer = new CUdeviceptr();

        JCudaDriver.cuMemAlloc(this.pointer, this.sizeBytes);

        if (input != null) {
            upload(input);
        }
    }

    public Tensor(int rows, int columns) {
        this(rows, columns, null);
    }

    public void upload(float[] input) {
        Pointer inputPointer = Pointer.to(input);
        JCudaDriver.cuMemcpyHtoD(this.pointer, inputPointer, this.sizeBytes);
    }

    public void download(float[] output) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(output), this.pointer, this.sizeBytes);
    }

    public float[] download() {
        float[] output = new float[this.size];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(output), this.pointer, this.sizeBytes);
        return output;
    }

    public void close() {
        JCudaDriver.cuMemFree(this.pointer);
    }

    public CUdeviceptr getPointer() {
        return this.pointer;
    }

    public int getRows() {
        return this.rows;
    }

    public int getColumns() {
        return this.columns;
    }

    public int getSize() {
        return this.size;
    }

    public int getSizeBytes() {
        return this.sizeBytes;
    }
}
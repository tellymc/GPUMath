package me.tellymc;

import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import me.tellymc.objects.Tensor;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class GPUMath {

    private static final String FILE_NAME = "kernel";
    private static CUmodule module;
    private static CUfunction exponentialFunction;
    private static CUfunction powerScalarFunction;
    private static CUfunction elementwiseMultiplyFunction;
    private static CUfunction scaleRowsFunction;
    private static CUfunction scaleColumnsFunction;
    private static cublasHandle handle;

    public static void init() {
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        module = new CUmodule();

        try (InputStream is = GPUMath.class.getResourceAsStream("/" + FILE_NAME + ".fatbin")) {
            if (is == null) {
                throw new RuntimeException(FILE_NAME + ".fatbin not found in resources!");
            }
            File tempPtx = File.createTempFile(FILE_NAME, ".fatbin");
            tempPtx.deleteOnExit();
            Files.copy(is, tempPtx.toPath(), StandardCopyOption.REPLACE_EXISTING);

            JCudaDriver.cuModuleLoad(module, tempPtx.getAbsolutePath());
        } catch (Exception e) {
            throw new RuntimeException("Failed to load FatBin file from resources", e);
        }

        exponentialFunction = new CUfunction();
        powerScalarFunction = new CUfunction();
        elementwiseMultiplyFunction = new CUfunction();
        scaleRowsFunction = new CUfunction();
        scaleColumnsFunction = new CUfunction();

        JCudaDriver.cuModuleGetFunction(exponentialFunction, module, "exponential");
        JCudaDriver.cuModuleGetFunction(powerScalarFunction, module, "powerScalar");
        JCudaDriver.cuModuleGetFunction(elementwiseMultiplyFunction, module, "elementwiseMultiply");
        JCudaDriver.cuModuleGetFunction(scaleRowsFunction, module, "scaleRows");
        JCudaDriver.cuModuleGetFunction(scaleColumnsFunction, module, "scaleColumns");

        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
    }

    public static void initialize() {
        init();
    }

    public static void exponential(Tensor input, Tensor output) {
        Pointer parameters = Pointer.to(Pointer.to(input.getPointer()), Pointer.to(output.getPointer()), Pointer.to(new int[]{input.getSize()}));
        launchFunction(exponentialFunction, input.getSize(), parameters);
    }

    public static void powerScalar(Tensor input, float scalar, Tensor output) {
        Pointer parameters = Pointer.to(Pointer.to(input.getPointer()), Pointer.to(new float[]{scalar}), Pointer.to(output.getPointer()), Pointer.to(new int[]{input.getSize()}));
        launchFunction(powerScalarFunction, input.getSize(), parameters);
    }

    public static void elementwiseMultiply(Tensor matrixA, Tensor matrixB, Tensor output) {
        Pointer parameters = Pointer.to(Pointer.to(matrixA.getPointer()), Pointer.to(matrixB.getPointer()), Pointer.to(output.getPointer()), Pointer.to(new int[]{matrixA.getSize()}));
        launchFunction(elementwiseMultiplyFunction, matrixA.getSize(), parameters);
    }

    public static void scaleRows(Tensor matrixA, Tensor matrixB, Tensor output) {
        Pointer parameters = Pointer.to(Pointer.to(matrixA.getPointer()), Pointer.to(matrixB.getPointer()), Pointer.to(output.getPointer()), Pointer.to(new int[]{matrixA.getRows()}), Pointer.to(new int[]{matrixA.getColumns()}));
        launchFunction(scaleRowsFunction, matrixA.getSize(), parameters);
    }

    public static void scaleColumns(Tensor matrixA, Tensor matrixB, Tensor output) {
        Pointer parameters = Pointer.to(Pointer.to(matrixA.getPointer()), Pointer.to(matrixB.getPointer()), Pointer.to(output.getPointer()), Pointer.to(new int[]{matrixA.getRows()}), Pointer.to(new int[]{matrixA.getColumns()}));
        launchFunction(scaleColumnsFunction, matrixA.getSize(), parameters);
    }

    public static void multiply(Tensor matrixA, Tensor matrixB, boolean transposeA, boolean transposeB, Tensor output) {
        int m = matrixA.getRows();
        int k = matrixA.getColumns();
        int n = matrixB.getColumns();

        Pointer alpha = Pointer.to(new float[]{1.0f});
        Pointer beta = Pointer.to(new float[]{0.0f});

        int operationA = transposeA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
        int operationB = transposeB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;

        JCublas2.cublasSgemm(handle, operationB, operationA, n, m, k, alpha, matrixB.getPointer(), n, matrixA.getPointer(), k, beta, output.getPointer(), n);
    }

    public static void multiply(Tensor matrixA, Tensor matrixB, Tensor output) {
        multiply(matrixA, matrixB, false, false, output);
    }

    private static void launchFunction(CUfunction function, int size, Pointer parameters) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        JCudaDriver.cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, null, parameters, null);
    }
}
# GPU Math
[![](https://jitpack.io/v/tellymc/GPUMath.svg)](https://jitpack.io/#tellymc/GPUMath)

## How to Install

1) Add the Repository within the Pom.xml
```xml
<repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
</repository>
```

2) Add the Dependency within the Pom.xml
```xml
<dependency>
    <groupId>com.github.tellymc</groupId>
    <artifactId>GPUMath</artifactId>
    <version>{Version right under title}</version>
</dependency>
```

## What is  Tensor (Tensor vs Matrix)
##### A matrix and a tensor are very similar but there is a clear distinction between the two that is very important to know. They are both a matrix (technically), but a tensor is just saved within the VRAM (on the graphics card), which has a saved address/pointer in the RAM. To use a Tensor you will have to send commands, with the specific Tensors, to perform math operations on the Graphics Card instead of the Processor.

## Setting it up
```java
public static void main(String[] args) {

    // You must first initialize GPUMath within your project when it runs, or else the operations will NOT work
    // To initialize it is as simple as:
    GPUMath.init();

    // To create a Tensor, you just have to specify the rows and columns (dimensions) of your matrix
    Tensor matrixA = new Tensor(2, 2);
    Tensor matrixB = new Tensor(2, 2);

    // Fill the matrices by uploading data
    float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] dataB = {4.0f, 3.0f, 2.0f, 1.0f};

    matrixB.upload(dataA);
    matrixB.upload(dataB);

    // To retrieve the data is similar, just with .download()
    float[] resultA = new float[dataA.length];
    float[] resultB = new float[dataB.length];

    resultA = matrixA.download();
    resultB = matrixB.download();

    // IMPORTANT! To avoid having memory leaks, close your Tensors after using them
    matrixA.close();
    matrixB.close();
}
``` 

## Each of the Math Operation Methods

### Matrix Multiplication (The Dot Product)
```java
public class Main {

    public static void main(String[] args) {

        GPUMath.init();

        // Create your two matrices with your output
        Tensor matrixA = new Tensor(2, 3);
        Tensor matrixB = new Tensor(3, 2);
        Tensor output = new Tensor(2, 2);

        float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] dataB = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        matrixA.upload(dataA);
        matrixB.upload(dataB);

        // Using .multiply(), you can multiply the matrices
        GPUMath.multiply(matrixA, matrixB, output);
        
        // If you wish to transpose (flip the rows and columns), you can add 2 parameters
        // The first one transposing matrixA, the second transposing matrixB
        GPUMath.multiply(matrixA, matrixB, false, false, output);

        // Retrieve the output of the multiplication
        float[] result = output.download();
        
        System.out.println("The result: " + Arrays.toString(result));

        // IMPORTANT! To avoid having memory leaks, close your Tensors after using them
        matrixA.close();
        matrixB.close();
        output.close();
    }
}
``` 

### Exponential (e^x)
```java
public class Main {

    public static void main(String[] args) {

        GPUMath.init();

        // Create your matrix and output
        Tensor matrixA = new Tensor(2, 3);
        Tensor output = new Tensor(2, 3);

        float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        matrixA.upload(dataA);

        // Do the exponential of each value in the matrix, this will put the value into e^x
        GPUMath.exponential(matrixA, output);

        // Retrieve the output
        float[] result = output.download();

        System.out.println("The result: " + Arrays.toString(result));

        // IMPORTANT! To avoid having memory leaks, close your Tensors after using them
        matrixA.close();
        output.close();
    }
}
``` 

### Power Scalar (x^y)
```java
Coming soon...
``` 

### Elementwise Multiplication
```java
Coming soon...
``` 

### Scale Rows
```java
Coming soon...
``` 

### Scale Columns
```java
Coming soon...
```
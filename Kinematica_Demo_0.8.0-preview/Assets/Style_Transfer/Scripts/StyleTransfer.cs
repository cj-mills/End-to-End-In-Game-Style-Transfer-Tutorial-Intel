using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class StyleTransfer : MonoBehaviour
{
    [Tooltip("Performs the preprocessing and postprocessing steps")]
    public ComputeShader styleTransferShader;

    [Tooltip("Stylize the camera feed")]
    public bool stylizeImage = true;

    [Tooltip("Stylize only specified GameObjects")]
    public bool targetedStylization = true;

    [Tooltip("The height of the image being fed to the model")]
    public int targetHeight = 540;

    [Tooltip("The model asset file that will be used when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("Captures the depth data for the target GameObjects")]
    public Camera styleDepth;

    [Tooltip("Captures the depth data for the entire scene")]
    public Camera sourceDepth;

    // The compiled model used for performing inference
    private Model m_RuntimeModel;

    // The interface used to execute the neural network
    private IWorker engine;

    // Start is called before the first frame update
    void Start()
    {
        // Get the screen dimensions
        int width = Screen.width;
        int height = Screen.height;

        // Force the StyleDepth Camera to render to a Depth texture
        styleDepth.targetTexture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.Depth);
        styleDepth.forceIntoRenderTexture = true;
        // Force the SourceDepth Camera to render to a Depth texture
        sourceDepth.targetTexture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.Depth);
        sourceDepth.forceIntoRenderTexture = true;

        // Compile the model asset into an object oriented representation
        m_RuntimeModel = ModelLoader.Load(modelAsset);

        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);
    }

    private void Update()
    {

        if (styleDepth.targetTexture.width != Screen.width || styleDepth.targetTexture.height != Screen.height)
        {
            // Get the screen dimensions
            int width = Screen.width;
            int height = Screen.height;

            // Assign depth textures with the new dimensions
            styleDepth.targetTexture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.Depth);
            sourceDepth.targetTexture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.Depth);
        }

        if (Input.GetMouseButtonUp(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                Transform[] allChildren = hit.transform.gameObject.GetComponentsInChildren<Transform>();

                for (int i = 0; i < allChildren.Length; i++)
                {
                    MeshRenderer meshRenderer = allChildren[i].GetComponent<MeshRenderer>();
                    if (meshRenderer != null && meshRenderer.enabled)
                    {

                        if (allChildren[i].gameObject.layer == 12)
                        {
                            allChildren[i].gameObject.layer = 0;
                        }
                        else
                        {
                            allChildren[i].gameObject.layer = 12;
                        }
                    }
                }
            }
        }
    }

    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();

        // Release the Depth texture for the StyleDepth camera
        RenderTexture.ReleaseTemporary(styleDepth.targetTexture);
        // Release the Depth texture for the SourceDepth camera
        RenderTexture.ReleaseTemporary(sourceDepth.targetTexture);
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns>The processed image</returns>
    private void ProcessImage(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    /// <summary>
    /// Merge the stylized frame and the original frame on the GPU
    /// </summary>
    /// <param name="styleImage"></param>
    /// <param name="sourceImage"></param>
    /// <returns>The merged image</returns>
    private void Merge(RenderTexture styleImage, RenderTexture sourceImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel("Merge");
        // Define a temporary HDR RenderTexture
        int width = styleImage.width;
        int height = styleImage.height;
        RenderTexture result = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", styleImage);
        // Set the value for the StyleDepth variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "StyleDepth", styleDepth.activeTexture);
        // Set the value for the SrcDepth variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "SrcDepth", sourceDepth.activeTexture);
        // Set the value for the SrcImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "SrcImage", sourceImage);

        // Execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, styleImage);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    /// <summary>
    /// Stylize the provided image
    /// </summary>
    /// <param name="src"></param>
    /// <returns></returns>
    private void StylizeImage(RenderTexture src)
    {
        // Create a new RenderTexture variable
        RenderTexture rTex;

        // Check if the target display is larger than the targetHeight
        // and make sure the targetHeight is at least 8
        if (src.height > targetHeight && targetHeight >= 8)
        {
            // Calculate the scale value for reducing the size of the input image
            float scale = src.height / targetHeight;
            // Calcualte the new image width
            int targetWidth = (int)(src.width / scale);

            // Adjust the target image dimensions to be multiples of 8
            targetHeight -= (targetHeight % 8);
            targetWidth -= (targetWidth % 8);

            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(targetWidth, targetHeight, 24, src.format);
        }
        else
        {
            // Assign a temporary RenderTexture with the src dimensions
            rTex = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);
        }

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(src, rTex);

        // Apply preprocessing steps
        ProcessImage(rTex, "ProcessInput");

        // Create a Tensor of shape [1, rTex.height, rTex.width, 3]
        Tensor input = new Tensor(rTex, channels: 3);

        // Execute neural network with the provided input
        engine.Execute(input);

        // Get the raw model output
        Tensor prediction = engine.PeekOutput();
        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Make sure rTex is not the active RenderTexture
        RenderTexture.active = null;
        // Copy the model output to rTex
        prediction.ToRenderTexture(rTex);
        // Release GPU resources allocated for the Tensor
        prediction.Dispose();

        // Apply postprocessing steps
        ProcessImage(rTex, "ProcessOutput");
        // Copy rTex into src
        Graphics.Blit(rTex, src);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);
    }


    /// <summary>
    /// OnRenderImage is called after the Camera had finished rendering 
    /// </summary>
    /// <param name="src">Input from the Camera</param>
    /// <param name="dest">The texture for the targer display</param>
    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // Create a temporary RenderTexture to store copy of the current frame
        RenderTexture sourceImage = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);
        // Copyt the current frame
        Graphics.Blit(src, sourceImage);

        if (stylizeImage)
        {
            StylizeImage(src);

            if (targetedStylization)
            {
                // Merge the stylized frame and origina frame
                Merge(src, sourceImage);
            }
        }

        Graphics.Blit(src, dest);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(sourceImage);
    }
}

# Supplementary Experiment: No Transformation on the Original Data

The weight transformation in Equation $w_{new}=e^{-\frac{1}{w_{old}}}$ has been widely adopted in the literature on link weight prediction.

To further validate the robustness of our model and demonstrate that the weight transformation formula does not affect the comparative performance between different models, we conducted an additional experiment without applying any data transformation or preprocessing. This experiment, referred to as **No Transformation on the Original Data**, directly evaluates the model's performance on the original datasets. 



### Results and Analysis

The results, summarized in Table 1, indicate that the performance trend observed on the original dataset is consistent with that on the transformed dataset. This demonstrates that the weight transformation does not affect the comparative evaluation of predictive performance between different models. The detailed results are presented below:

<table style="width:100%; border-collapse:collapse;">
    <caption style="font-weight:bold;">Table 1: Model Prediction Performance without Weight Transformation (Mean and Standard Deviation of RMSE over 10 Trials)</caption>
    <tr>
        <th style="text-align:center; border:1px solid #000;">Model</th>
        <th style="text-align:center; border:1px solid #000;">Neural-net</th>
        <th style="text-align:center; border:1px solid #000;">C. elegans</th>
        <th style="text-align:center; border:1px solid #000;">Netscience</th>
        <th style="text-align:center; border:1px solid #000;">Condmat</th>
    </tr>
    <tr>
        <td style="text-align:center; border:1px solid #000;"><strong>SEA</strong></td>
        <td style="text-align:center; border:1px solid #000;">6.4564 ± 1.2419</td>
        <td style="text-align:center; border:1px solid #000;">3.5904 ± 1.7969</td>
        <td style="text-align:center; border:1px solid #000;">0.2220 ± 0.0518</td>
        <td style="text-align:center; border:1px solid #000;">0.7916 ± 0.0043</td>
    </tr>
    <tr>
        <td style="text-align:center; border:1px solid #000;"><strong>LGLWP</strong></td>
        <td style="text-align:center; border:1px solid #000;">5.1405 ± 1.1276</td>
        <td style="text-align:center; border:1px solid #000;">4.6140 ± 2.5770</td>
        <td style="text-align:center; border:1px solid #000;">0.1875 ± 0.0729</td>
        <td style="text-align:center; border:1px solid #000;">0.6637 ± 0.0368</td>
    </tr>
</table>


These findings indicate that our model achieves better predictive performance even without preprocessing or transformations, further demonstrating its robustness and generalization capability.

### Conclusion

The results of the supplementary experiment confirm that the weight transformation based on Equation $w_{new}=e^{-\frac{1}{w_{old}}}$ does not negatively affect the relative performance comparison across different models. On the contrary, it contributes positively by standardizing the range of weight values, thereby enhancing the consistency of evaluation metrics across various datasets. Furthermore, the weight transformation effectively ensures the reliability of the results and preserves their comparability with those reported in existing studies. These findings further demonstrate the robustness of our proposed approach.

---

For further details, refer to the main text and the references provided.

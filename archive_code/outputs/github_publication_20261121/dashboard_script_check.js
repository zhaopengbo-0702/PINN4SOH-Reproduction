
    (() => {
      'use strict';
      const imageRoot = 'https://raw.githubusercontent.com/zhaopengbo-0702/PINN4SOH-Reproduction/main/outputs/full_reproduction/plot_final2120_complete_v4/figures/';
      const lanes = {
        published: {
          title: '标签单调实现：多数常规组误差更低，MIT/HUST 保留反例',
          description: '11 组的三项误差中，27/33 个组指标同时低于 MLP 与 CNN；MIT/HUST 未形成这一均值优势。',
          alt: '标签单调 PINN、MLP 与 CNN 在 11 组常规任务中的 MAE、MAPE、RMSE 十次均值及样本标准差'
        },
        paperloss: {
          title: '预测单调实现：结果显著不同，不能与标签实现合并',
          description: '在同一主实验比较范围，0/33 个常规组指标同时低于两基线。实现差异会影响结论，较小重复波动也不等于较小平均误差。',
          alt: '预测单调 PINN、MLP 与 CNN 在 11 组常规任务中的 MAE、MAPE、RMSE 十次均值及样本标准差'
        }
      };
      document.querySelectorAll('[data-lane]').forEach(button => {
        button.addEventListener('click', () => {
          const lane = button.dataset.lane;
          if (!Object.prototype.hasOwnProperty.call(lanes, lane)) return;
          document.querySelectorAll('[data-lane]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
          const imageUrl = imageRoot + 'Fig4b_' + lane + '.png';
          document.getElementById('chart-title').textContent = lanes[lane].title;
          document.getElementById('chart-description').textContent = lanes[lane].description;
          const chart = document.getElementById('regular-chart');
          chart.src = imageUrl;
          chart.alt = lanes[lane].alt;
          document.getElementById('chart-original').href = imageUrl;
          document.getElementById('chart-open').href = imageUrl;
        });
      });
      const metricRows = {
        pooled: {
          description: '先在每次运行内合并全部测试样本计算指标，再对 10 次结果求均值与样本标准差。较长测试文件的样本权重更大。',
          mit: ['0.689402', '0.028627', '0.902220', '0.042616'],
          hust: ['0.814535', '0.027752', '0.955581', '0.029957']
        },
        macro: {
          description: '先在每次运行内分别计算各测试 CSV 的指标，再对 CSV 等权平均，最后统计 10 次结果。MIT 混合物理身份文件的限制仍保留。',
          mit: ['0.792423', '0.050746', '0.872649', '0.047542'],
          hust: ['0.799301', '0.030026', '0.893679', '0.037698']
        }
      };
      document.querySelectorAll('[data-metric]').forEach(button => {
        button.addEventListener('click', () => {
          const row = metricRows[button.dataset.metric];
          if (!row) return;
          document.querySelectorAll('[data-metric]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
          document.getElementById('aggregation-description').textContent = row.description;
          ['mit', 'hust'].forEach(dataset => {
            const values = row[dataset];
            document.getElementById(dataset + '-mape').textContent = values[0];
            document.getElementById(dataset + '-mape-sd').textContent = ' ± ' + values[1];
            document.getElementById(dataset + '-rmse').textContent = values[2];
            document.getElementById(dataset + '-rmse-sd').textContent = ' ± ' + values[3];
          });
        });
      });
    })();
  
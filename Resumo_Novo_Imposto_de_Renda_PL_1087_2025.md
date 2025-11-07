# Resumo técnico do **Projeto de Lei 1.087/2025** — *Novo Imposto de Renda*

Documento preparado para alimentar um agente de IA (assistente financeiro).  
Contém: resumo executivo e detalhado das mudanças, estrutura de **Tabela de Casos** para validação e **funções de cálculo (tools)** prontas para uso pelo agente.

---

## 1. Resumo Executivo

O Projeto de Lei nº 1.087/2025 altera o Imposto de Renda da Pessoa Física (IRPF) para:

1. **Reduzir o imposto devido** nas bases de cálculo **mensal e anual**, ampliando a faixa de isenção para rendas baixas e médias.
2. **Instituir a Tributação Mínima sobre Altas Rendas (IRPFM)**, incidindo sobre contribuintes com rendimentos totais anuais superiores a R$ 600 mil.
3. **Estabelecer retenção antecipada** de 10% sobre lucros e dividendos mensais acima de R$ 50 mil pagos pela mesma fonte.
4. **Criar redutor/crédito** para evitar dupla tributação entre empresa e pessoa física.

**Entrada em vigor:** 1º de janeiro de 2026.

**Objetivo:** aumentar a progressividade do IR, aliviar a carga sobre trabalhadores assalariados e promover justiça tributária sobre rendas de capital e altas fortunas.

---

## 2. Principais Mudanças Detalhadas

### 2.1 Redução do Imposto Mensal (a partir de janeiro de 2026)

| Faixa de Rendimento Tributável Mensal (R$) | Fórmula de Redução | Observação |
|--------------------------------------------|--------------------|-------------|
| Até 5.000,00 | Redução fixa de **R$ 312,89** | Imposto mensal = 0 |
| De 5.000,01 a 7.000,00 | **Redução = 1.095,11 − (0,156445 × renda_mensal)** | Redução decrescente |
| Acima de 7.000,00 | Redução = 0 | Sem benefício |

- O valor da redução **não pode exceder** o imposto apurado pela tabela progressiva mensal.  
- A redução também se aplica ao cálculo do imposto retido sobre o **13º salário**.  
- Gestão: Secretaria de Política Econômica do Ministério da Fazenda.

---

### 2.2 Redução do Imposto Anual (exercício 2027 — ano-calendário 2026)

| Faixa de Rendimento Tributável Anual (R$) | Fórmula de Redução | Observação |
|-------------------------------------------|--------------------|-------------|
| Até 60.000,00 | Redução fixa de **R$ 2.694,15** | Imposto anual = 0 |
| De 60.000,01 a 84.000,00 | **Redução = 9.429,52 − (0,1122562 × renda_anual)** | Redução decrescente |
| Acima de 84.000,00 | Redução = 0 | Sem benefício |

- A redução é aplicada no cálculo do ajuste anual da declaração.  
- Os valores são fixos e independem do regime de desconto simplificado ou completo.  

---

### 2.3 Tributação Mínima sobre Altas Rendas (IRPFM)

- Incide sobre pessoas físicas com **rendimentos totais superiores a R$ 600.000,00/ano**.
- Base de cálculo: soma de todos os rendimentos (inclusive isentos e tributados exclusivamente na fonte), exceto:
  - Ganhos de capital;
  - Rendimentos acumulados com tributação exclusiva;
  - Doações e heranças.

**Alíquotas progressivas:**

| Faixa de Rendimento Anual | Alíquota IRPFM |
|----------------------------|----------------|
| Até R$ 600.000,00 | 0% |
| De R$ 600.000,01 a R$ 1.200.000,00 | `(renda / 60.000) − 10` |
| Acima de R$ 1.200.000,00 | 10% |

**Cálculo:**
```
IRPFM devido = (base * alíquota) – deduções legais – IR pago – IR retido – IR definitivo – redutor
```

Se o resultado for negativo → imposto devido = 0.

---

### 2.4 Retenção Mensal sobre Lucros e Dividendos

- Quando uma mesma **pessoa jurídica** pagar, creditar ou entregar **mais de R$ 50.000,00** em lucros/dividendos a **uma mesma pessoa física** no mês:
  - **Retenção de 10%** sobre o total pago.
- A retenção **não admite deduções** e **funciona como antecipação** do IRPFM anual.
- O agente deve somar todos os pagamentos da mesma empresa (CNPJ) no mesmo mês antes de aplicar a retenção.

---

### 2.5 Redutor / Crédito para Evitar Dupla Tributação

- Se a soma da carga tributária da empresa (IRPJ + CSLL) **somada à** alíquota efetiva do IRPFM ultrapassar as alíquotas nominais máximas previstas (34% a 45%), o contribuinte poderá aplicar:
  - **Redutor (pessoa física residente)** ou  
  - **Crédito (beneficiário no exterior)**.

**Fórmula simplificada:**
```
Redutor = montante_dividendos × ((alíquota_empresa + alíquota_irpfm) − alíquota_nominal_limite)
```

- O redutor é limitado a zero se o resultado for negativo.  
- Exige demonstrações financeiras da empresa ou aplicação simplificada.

---

### 2.6 Outras Alterações Relevantes

- Lucros e dividendos pagos a **não residentes** no Brasil terão **retenção de 10%** na fonte, com mecanismo de crédito para evitar dupla tributação.  
- O IRPFM será apurado no **ajuste anual** da declaração, compensando as retenções já feitas.  
- O benefício da redução mensal e anual está vinculado a rendimentos provenientes do trabalho ou aposentadoria.

---

## 3. Vigência e Impactos

- **Vigência:** 1º de janeiro de 2026.  
- **Beneficiados:** trabalhadores e aposentados com renda até R$ 5.000 mensais (isenção total).  
- **Tributados adicionalmente:** pessoas físicas com rendimentos totais anuais acima de R$ 600 mil.  
- O governo projeta **renúncia fiscal** compensada por aumento da arrecadação sobre dividendos e altas rendas.  

---

## 4. Estrutura da Tabela de Casos (Template para Validação do Agente)

| case_id | scenario_title | description | period | monthly_taxable_income | annual_taxable_income | uses_simplified_discount | dividends_payments | corporate_effective_tax_rate | company_profit_contabil | expected_monthly_reduction | expected_monthly_tax_after_reduction | expected_withholding_on_dividends | expected_irpfm_annual | expected_redutor_or_credit | expected_final_tax_balance | notes_for_agent | validation_status | reference_files |
|----------|----------------|--------------|---------|------------------------|----------------------|---------------------------|--------------------|-------------------------------|--------------------------|-----------------------------|-------------------------------------|----------------------------------|----------------------|-----------------------------|---------------------------|-----------------|------------------|----------------|

---

## 5. Funções de Cálculo (para integração como Tools do Agente)

### 5.1 Redução Mensal
```
if renda <= 5000:
    reducao = 312.89
elif renda <= 7000:
    reducao = 1095.11 - (0.156445 * renda)
else:
    reducao = 0
reducao = min(reducao, imposto_progressivo_mensal)
```

### 5.2 Redução Anual
```
if renda <= 60000:
    reducao = 2694.15
elif renda <= 84000:
    reducao = 9429.52 - (0.1122562 * renda)
else:
    reducao = 0
reducao = min(reducao, imposto_progressivo_anual)
```

### 5.3 Retenção sobre Dividendos
```
if total_dividendos_por_cnpj_mes > 50000:
    retencao = total * 0.10
else:
    retencao = 0
```

### 5.4 Alíquota IRPFM Anual
```
if renda >= 1200000:
    aliquota = 10
elif renda <= 600000:
    aliquota = 0
else:
    aliquota = (renda / 60000) - 10
```

### 5.5 IRPFM Básico (sem redutor)
```
base = renda - deducoes
irpfm = base * (aliquota / 100)
irpfm -= (ir_pago + retido + definitivo)
if irpfm < 0:
    irpfm = 0
```

### 5.6 Redutor (para evitar dupla tributação)
```
diff = (aliquota_empresa + aliquota_irpfm) - aliquota_nominal_limite
if diff > 0:
    redutor = dividendos * diff
else:
    redutor = 0
```

---

## 6. Checklist de Validação

1. Agrupar pagamentos de dividendos **por CNPJ e mês** antes de aplicar retenção.  
2. Incluir rendimentos isentos/exclusivos na base do IRPFM (exceto os excluídos pelo PL).  
3. Garantir que a redução mensal/anual **não exceda o imposto apurado**.  
4. Aplicar redutor apenas com **demonstrações financeiras válidas**.  
5. Tratar todas as retenções mensais como **antecipações** do IRPFM anual.  

---

## 7. Observações Finais

- O agente deve manter **rastreabilidade completa** de todos os cálculos e dados de entrada.  
- Para **beneficiários no exterior**, aplicar o crédito tributário previsto no art. 10-A.  
- O **redutor** depende de dados contábeis da empresa (lucro, IRPJ, CSLL).  
- Recomenda-se armazenar logs de cálculo com data, base legal e versão da tabela progressiva.

---

## 8. Referências Legais (PL 1.087/2025)

- **Art. 3º-A** — Redução do imposto mensal.  
- **Art. 11-A** — Redução do imposto anual.  
- **Art. 6º-A** — Retenção de 10% sobre lucros/dividendos mensais acima de R$ 50.000,00.  
- **Art. 16-A** — Definição e cálculo do IRPFM.  
- **Art. 16-B** — Redutor/crédito de dupla tributação.  
- **Art. 10-A** — Tributação de lucros remetidos ao exterior.  

---

### Versão: 1.0 — Baseada em PL 1.087/2025 (texto integral)
### Produzido em: Novembro de 2025
### Fonte: Governo Federal / Planalto / Ministério da Fazenda

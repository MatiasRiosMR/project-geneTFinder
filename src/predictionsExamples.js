// script que crea botones de ejemplo y rellena el input.
// Es independiente y tolerante si la estructura del DOM difiere; solo necesita los ids del input y del contenedor.

// Añade botones de ejemplo en la ventana de predicción.
// Uso: incluir el script en la página y llamar:
// initExampleButtons('prediction-input', 'example-buttons-container');
function initExampleButtons(predictionInputId, buttonsContainerId) {
	const input = document.getElementById(predictionInputId);
	const container = document.getElementById(buttonsContainerId);
	if (!input || !container) return;

	// Ruta relativa al JSON; ajustar si su servidor expone otra ruta.
	const dataPath = '/data/predictions.json';

	fetch(dataPath)
		.then((res) => res.json())
		.then((data) => {
			const examples = (data && data.examples) || {
				TF: { sequence: 'MGRKKIQITRIMDERNRQVTFTKRKFGLMKKAYELSVLCDCEI', label: 'Factor de Transcripción (TF)' },
				'No-TF': { sequence: 'AAAAATTTAAATTAAAGGGACACAGAACATAGACAGTACGTAGG', label: 'No-TF' }
			};

			['TF', 'No-TF'].forEach((key) => {
				const btn = document.createElement('button');
				btn.type = 'button';
				btn.textContent = `Ejemplo: ${key}`;
				btn.className = `example-btn example-${key.replace(/\W/g, '')}`;
				btn.addEventListener('click', () => {
					input.value = examples[key].sequence || '';
					// Si hay una función global predict(), la llamamos para ejecutar la predicción automáticamente.
					if (typeof window.predict === 'function') {
						try { window.predict(); } catch (e) { /* silencioso */ }
					}
				});
				container.appendChild(btn);
			});
		})
		.catch(() => {
			// Fallback: crear dos botones con ejemplos sencillos
			const fallback = {
				TF: 'MGRKKIQITRIMDERNRQVTFTKRKFGLMKKAYELSVLCDCEI',
				'No-TF': 'AAAAATTTAAATTAAAGGGACACAGAACATAGACAGTACGTAGG'
			};
			['TF', 'No-TF'].forEach((key) => {
				const btn = document.createElement('button');
				btn.type = 'button';
				btn.textContent = `Ejemplo: ${key}`;
				btn.addEventListener('click', () => {
					input.value = fallback[key];
					if (typeof window.predict === 'function') {
						try { window.predict(); } catch (e) { /* silencioso */ }
					}
				});
				container.appendChild(btn);
			});
		});
}
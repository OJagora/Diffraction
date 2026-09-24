# Diffraction Studio

A static, browser based diffraction demo. Draw an aperture, choose a wavelength or spectrum, and see the far-field intensity pattern update as you work.

## Run locally

Serve the repository root with any static file server, then open /web/ in the browser. The worker is loaded relative to the web app, so keep the files together when deploying.

## Interaction

- Draw with a mouse, pen, or touch. Hold Shift or use the right mouse button to erase.
- Choose a single wavelength or a polychromatic spectrum from 380 to 780 nm.
- Compare 128, 256, and 512 pixel aperture resolutions.
- Load the round, double slit, or JWST-inspired aperture presets.
- Clear or reset the current aperture and controls.

## How the live rendering works

The UI samples the aperture and sends a transferable copy to a Web Worker. The worker computes one 2D Fourier transform, then maps the reference intensity pattern to each selected wavelength before composing the colour result. This keeps the browser's main thread available for drawing and control input, including at the highest resolution.

The simulation uses a scalar Fraunhofer model with logarithmic intensity display. The displayed spectrum uses evenly spaced wavelength samples and an approximate wavelength-to-RGB mapping; it is intended as an interactive educational visualization rather than a calibrated optical instrument.

The aperture state can also be read or restored through window.diffractionApp.getApertureState() and window.diffractionApp.setApertureState(state). The version 1 state contains a resolution and a flat binary mask array.

The app is self-contained and has no backend or build step. Deploy the contents of web/ under a path such as /diffraction/ and preserve the relative paths to app.js, diffraction-worker.js, and styles.css.
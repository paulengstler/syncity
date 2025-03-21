import { Link as Highlight, Text, Code, useDisclosure, ListItem, Hide, Heading, Collapse, useColorModeValue, Container, Show, Button, Box, Link, Stack, OrderedList, UnorderedList, AspectRatio, Table, Tr, Td, Tbody } from '@chakra-ui/react'
import { ExternalLinkIcon, ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons'
import { ArticleHeading } from '@/components/ArticleHeading'
import { LinksRow } from 'components/LinksRow'
import React, { useState, useEffect } from 'react';

import { title, authors, acknowledgements, author_tags } from 'data'

import { useColorMode, useColorModePreference } from '@chakra-ui/react'

function renderViewLarger(url: string, description?: string, renderLinkOnly: boolean = false) {
  const brandSwitch = useColorModeValue('brand.600', 'brand.500')

  return renderLinkOnly ? (
    <>
    <Link href={url} color={brandSwitch} target="_blank" fontFamily={"IBM Plex Sans"}>View a larger version{description !== undefined ? " of this " + description : ""} <ExternalLinkIcon /></Link>
    </>
  ) : (
    <>
    <Text align="left" pt="0.5rem" fontSize="small" fontFamily={"IBM Plex Sans"}><Link href={url} color={brandSwitch} target="_blank">View a larger version{description !== undefined ? " of this " + description : ""} <ExternalLinkIcon /></Link></Text>
    </>
  )
}

function toggleColorToMatchSystem() {
  const { colorMode, toggleColorMode } = useColorMode()
  const systemColorMode = useColorModePreference()

  if (colorMode !== systemColorMode && systemColorMode !== undefined) {
    toggleColorMode()
  }
}

function renderMoreResultsToggle() {
  const { isOpen, onToggle } = useDisclosure();

  function renderButton(isOpen: boolean) {
    if (isOpen) {
      return (
        <>
          <Button size='md' rightIcon={<ChevronUpIcon />}  isActive={isOpen} onClick={onToggle} fontFamily={"IBM Plex Sans"} mt="2rem">Show Less Results</Button>
        </>
      )
    } else {
      return (
        <>
          <Button size='md' rightIcon={<ChevronDownIcon />} onClick={onToggle} fontFamily={"IBM Plex Sans"} mt="2rem">Show More Results</Button>
        </>
      )
    }
  }

  return (
    <>
    {renderButton(isOpen)}
    <Collapse in={isOpen} animateOpacity>
      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_river_1_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>
    
      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_themepark_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>

      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_mayan_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>

      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_campus_rural_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>
    </Collapse>
    </>
  )
}

export const MoreResults = () => (
  renderMoreResultsToggle()
)

const Index = () => (
  <Box maxWidth="100%" backgroundColor={useColorModeValue('white', 'gray.700')}>
  <Container w="100%" maxWidth="4xl" backgroundColor={useColorModeValue('white', 'gray.700')} paddingStart={{ base: 'var(--chakra-space-0)', md: 'var(--chakra-space-4)'}} paddingEnd={{ base: 'var(--chakra-space-0)', md: 'var(--chakra-space-4)'}}>
    <Container w="100%" maxWidth="4xl" alignItems="left" pl="1.5rem" pr="1.5rem">

      {toggleColorToMatchSystem()}

      {/* Title */}
      <Heading fontSize={{ base: '3xl', md: '4xl' }} pt={{ base: '2.5vh', md: '5vh' }} pb={{ base: '1rem', md: '1rem' }} fontWeight="700">{title}</Heading>
      {/* <Authors /> */}
      <Box fontFamily={"IBM Plex Sans"}>{ authors.map((author, idx) => 
          <Box key={idx} display={'inline'} paddingRight={idx === authors.length - 1 ? '0' : '0.75rem'}><Link href={author.url} color={useColorModeValue('brand.600', '#fff')} fontWeight="600" target="_blank">{author.name}</Link>{author.tags.map((tag, tidx) => (
            <span>
              {tag}
              {tidx === author.tags.length - 1 ? "" : ", "}
            </span>
          ))}</Box>
        ) }</Box>
      <Text fontFamily={"IBM Plex Sans"}fontSize="sm" color={useColorModeValue('gray.600', 'gray.300')}>Visual Geometry Group, University of Oxford</Text>
      { author_tags.map((tag, idx) => (
        <Box key={idx} fontFamily={"IBM Plex Sans"} fontSize="xs" color={useColorModeValue('gray.600', 'gray.300')} mt=".5rem">
          {tag.name} denotes {tag.description}
        </Box>
      )) }

      <AspectRatio maxW='720px'>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)", marginTop: "1rem", objectFit: "contain"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/website_video_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>

      <LinksRow />

      <Box textAlign="justify">

      <ArticleHeading id="dataset" mt="0">Summary</ArticleHeading>
      <Text>
      SynCity generates complex and immersive 3D worlds from text prompts and does not require any training or optimization. It leverages the pretrained 2D image generator <Link href="https://blackforestlabs.ai" color={useColorModeValue('brand.600', '#fff')} target="_blank">Flux <ExternalLinkIcon style={{transform: "translateY(-2px)"}} /></Link> (for artistic diversity and contextual coherence) and the 3D generator <Link href="https://blackforestlabs.ai" color={useColorModeValue('brand.600', '#fff')} target="_blank">TRELLIS <ExternalLinkIcon style={{transform: "translateY(-2px)"}} /></Link> (for accurate geometry). We incrementally build scenes on a grid in a tile-by-tile fashion: First, we generate each tile as a 2D image, where context from adjacent tiles establishes consistency. Then, we convert the tile into a 3D model. Finally, adjacent tiles are blended seamlessly into a coherent, navigable 3D environment.
      </Text>
      <Box mt="1rem">
        <img src={`${process.env.BASE_PATH || ""}images/main-method.jpg`} alt="Method figure of SynCity"/>
        <Text fontSize="small" mt="1em">
          <span style={{fontWeight: "600"}}>Overview of SynCity. </span> 
          2D prompting: To generate a new tile, we first render a view of where that tile should be placed, including context from neighbouring tiles. 3D prompting: We extract the new tile image and construct an image prompt for TRELLIS by adding a wider base under the tile. 3D blending: The 3D model that TRELLIS outputs is usually not well blended with the rest of the scene. To address that,  we render a view of the new tile next to each neighbouring tile, and inpaint the region between the two with an image inpainting model. Next, we condition using that well-blended view to refine the region between the two 3D tiles. Finally, the new, blended, tile is added to the world.
        </Text>
      </Box>

      <ArticleHeading>Result Gallery</ArticleHeading>

  
      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_solarpunk_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>
    
      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_medieval_market_1_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>
    
      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_postapocalyptic_1_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>

      <AspectRatio maxW='720px' ratio={3 / 2}>
        <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
            <source src={`${process.env.BASE_PATH || ""}videos/render_campus_large_seasons_out.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </AspectRatio>

      <MoreResults />

      <ArticleHeading>Exploring Generated Worlds</ArticleHeading>
        <Text>
        The 3D worlds generated by SynCity can be fully explored. Here, we show some example trajectories that demonstrate the rich detail and immersive nature of the generations. A sky box has been added for visual effect.
        </Text>
      <Stack gap="10">
        <Table>
        <Tbody>
          <Tr>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_apocalypse_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_village_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
          </Tr>
          <Tr>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_mayan_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_themepark_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
          </Tr>
          <Tr>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_city_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
            <Td borderBottom="0">
              <AspectRatio maxW='720px' ratio={3 / 2}>
                <video playsInline autoPlay loop muted style={{clipPath: "inset(2px 2px)"}}>
                    <source src={`${process.env.BASE_PATH || ""}videos/walk_campus_across_seasons_out.mp4`} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
              </AspectRatio>
            </Td>
          </Tr>
          </Tbody>
        </Table>
      </Stack>

      <ArticleHeading id="interactive-demo">Interactive Demo</ArticleHeading>
      <Text>
      Explore a generated world in your browser! This interactive demo allows you to navigate through the generated scene and experience the immersive environment yourself.
      </Text>
      <br></br>
      <Show breakpoint="(max-width: 480px)">
        <Text>
          <i>The demo requires a screen width of at least 480px.</i>
        </Text>
      </Show>
      <Hide breakpoint="(max-width: 480px)">
      <iframe src={`${process.env.BASE_PATH || ""}demo/index.html`} width="480px" height="320px" frameBorder="0" allowFullScreen={true} allow="autoplay; fullscreen; xr-spatial-tracking"></iframe>
      </Hide>
      <br></br>
      <Text>
      Use WASD to move and QE to raise/lower the camera.
      </Text>
      <br></br>
      <Text fontSize="small">
      <i>Please note that the generated world has been compressed to enable a smooth in-browser experience. Thus, the visual quality is slightly degraded compared to the videos above. On devices without a dedicated GPU, the demo might appear choppy.</i>
      </Text>

      </Box>

      {/* Acknowledgements */}
      <Box pb="2rem">
      <ArticleHeading fontSize="md" color={useColorModeValue('gray.600', 'gray.300')}>Acknowledgements</ArticleHeading>
      <Text fontSize="small" color={useColorModeValue('gray.600', 'gray.300')}>
        {acknowledgements} The interactive demo is powered by the <Link href="https://playcanvas.com" target="_blank">PlayCanvas WebGL Engine <ExternalLinkIcon style={{transform: "translateY(-2px)"}} /></Link>.
      </Text>
      </Box>

  </Container>
  </Container>
  </Box>
)

export default Index